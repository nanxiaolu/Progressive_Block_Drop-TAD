import os
import sys
import shutil
sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import train_one_epoch_dist, val_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from opentad.utils import (
    set_seed,
    update_workdir,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_checkpoint,
    save_best_checkpoint,
)



def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument("--cfg_options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    if args.local_rank == 0:
        if os.path.isdir(cfg.work_dir):
            print('log_dir: ', cfg.work_dir, 'already exist')
            input("continue?")

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    cfg = update_workdir(cfg, args.id, args.world_size)
    if args.rank == 0:
        create_folder(cfg.work_dir)
        save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # tensorboard writer
    train_writer = SummaryWriter(os.path.join(cfg.work_dir, 'train'), 'train')

    # build dataset
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    train_loader = build_dataloader(
        train_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=True,
        drop_last=True,
        **cfg.solver.train,
    )

    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
    val_loader = build_dataloader(
        val_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.val,
    )

    best_mAP = 0.0
    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    test_loader = build_dataloader(
        test_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )

    # build model
    teacher_model = build_detector(cfg.teacher_model)
    student_model = build_detector(cfg.model)

    # DDP
    use_static_graph = getattr(cfg.solver, "static_graph", False)
    teacher_model = teacher_model.to(args.local_rank)
    teacher_model = DistributedDataParallel(teacher_model, device_ids=[args.local_rank], output_device=args.local_rank)

    student_model = student_model.to(args.local_rank)
    student_model = DistributedDataParallel(
        student_model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False if use_static_graph else True,
        static_graph=use_static_graph,  # default is False, should be true when use activation checkpointing in E2E
    )    
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    # FP16 compression
    use_fp16_compress = getattr(cfg.solver, "fp16_compress", False)

    if use_fp16_compress:
        logger.info("Using FP16 compression ...")
        teacher_model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
        student_model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        logger.info("Using Model EMA...")
        student_model_ema = ModelEma(student_model)
    else:
        student_model_ema = None

    # Load Trained Model
    device = f"cuda:{args.local_rank}"
    t_ckpt_path = cfg.teacher_model.backbone.custom['ckpt_path']
    logger.info(f"teacher model load ckpt from {t_ckpt_path}")
    t_checkpoint = torch.load(t_ckpt_path, map_location=device)
    if use_ema:
        teacher_model.module.load_state_dict(t_checkpoint["state_dict_ema"])
    else:
        teacher_model.module.load_state_dict(t_checkpoint["state_dict"])

    # Load student param from teacher
    s_ckpt_path = cfg.model.backbone.custom['ckpt_path']
    logger.info(f"student model load ckpt from {s_ckpt_path}")
    s_checkpoint = torch.load(s_ckpt_path, map_location=device)
    student_model.load_state_dict(s_checkpoint['state_dict'], strict=False)
    if use_ema:
        student_model_ema.module.load_state_dict(s_checkpoint['state_dict_ema'], strict=False)

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")
        scaler = GradScaler()
    else:
        scaler = None

    # build optimizer and scheduler
    optimizer = build_optimizer(cfg.optimizer, student_model, logger)
    warmup_epoch = cfg.scheduler.warmup_epoch
    scheduler, max_epoch = build_scheduler(cfg.scheduler, optimizer, len(train_loader))

    # override the max_epoch
    max_epoch = cfg.workflow.get("end_epoch", max_epoch)

    # resume: reset epoch, load checkpoint / best rmse
    if args.resume != None:
        logger.info("Resume training from: {}".format(args.resume))
        device = f"cuda:{args.local_rank}"
        checkpoint = torch.load(args.resume, map_location=device)
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        student_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if student_model_ema != None:
            student_model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

        del checkpoint  #  save memory if the model is very large such as ViT-g
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    # train the detector
    logger.info("Training Starts...\n")
    val_loss_best = 1e6
    val_first_start_epoch = cfg.workflow.get("val_first_start_epoch", 0)
    val_second_start_epoch = cfg.workflow.get("val_second_start_epoch", 0)
    for epoch in range(resume_epoch + 1, max_epoch):
        if cfg.model.backbone.backbone.type == 'VisionTransformerLongLoRAInplace':
            student_model.module.backbone.model.backbone.set_factor(epoch, max_epoch)

        if cfg.cal_loss.use_dkd_loss:
            cfg.solver.weight.loss_cls_kd = min(epoch / warmup_epoch, 1.0)

        train_loader.sampler.set_epoch(epoch)
        
        # train for one epoch
        train_one_epoch_dist(
            train_loader,
            teacher_model,
            student_model,
            optimizer,
            scheduler,
            epoch,
            logger,
            model_ema=student_model_ema,
            clip_grad_l2norm=cfg.solver.clip_grad_norm,
            logging_interval=cfg.workflow.logging_interval,
            scaler=scaler,
            train_writer=train_writer,
            weight=cfg.solver.weight,
        )

        # val for one epoch
        if (epoch + 1) >= val_second_start_epoch:
            if (cfg.workflow.val_loss_interval > 0) and ((epoch + 1) % cfg.workflow.val_loss_interval == 0):
                val_loss = val_one_epoch(
                    val_loader,
                    student_model,
                    logger,
                    args.rank,
                    epoch,
                    model_ema=student_model_ema,
                    use_amp=use_amp,
                )

                # save the best checkpoint
                if val_loss < val_loss_best:
                    logger.info(f"New best epoch {epoch}")
                    val_loss_best = val_loss
                    if args.rank == 0:
                        save_best_checkpoint(student_model, student_model_ema, epoch, work_dir=cfg.work_dir)

        # eval for one epoch
        eval_flag = False
        if (epoch + 1) >= val_second_start_epoch:
            if cfg.workflow.val_second_eval_interval > 0 and (epoch + 1) % cfg.workflow.val_second_eval_interval == 0:
                eval_flag = True
        elif (epoch + 1) >= val_first_start_epoch:
            if cfg.workflow.val_first_eval_interval > 0 and (epoch + 1) % cfg.workflow.val_first_eval_interval == 0:
                eval_flag = True

        if eval_flag:
            metrics_dict = eval_one_epoch(
                                test_loader,
                                student_model,
                                cfg,
                                logger,
                                args.rank,
                                model_ema=student_model_ema,
                                use_amp=use_amp,
                                world_size=args.world_size,
                                not_eval=args.not_eval,
                            )

            if metrics_dict:
                if best_mAP < metrics_dict["mAP@0.5"]:
                    best_mAP = metrics_dict["mAP@0.5"]
                    save_checkpoint(student_model, student_model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir)

    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()
