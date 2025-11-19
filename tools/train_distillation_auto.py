import os
import sys
import shutil
from copy import deepcopy
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
from torch.utils.data import Subset
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader, get_sub_dataloader_idx
from opentad.cores import (
    train_one_epoch_dist, 
    val_one_epoch, 
    eval_one_epoch, 
    build_optimizer, 
    build_scheduler, 
    eval_drop_one_block_mAP, 
    eval_drop_one_block_loss, 
    eval_drop_one_block_mse
)
from opentad.utils import (
    set_seed,
    update_workdir,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_val_checkpoint,
    save_test_checkpoint,
)
from tools.merge_model import MergeLoRA



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
    torch.set_num_threads(4)
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

    inference_train_dataset = build_dataset(cfg.dataset.inference_train, default_args=dict(logger=logger))
    inference_train_loader = build_dataloader(
        inference_train_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
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

    if 'sub_val' in cfg.dataset:
        sub_val_dataset = build_dataset(cfg.dataset.sub_val, default_args=dict(logger=logger))
        sub_val_loader = build_dataloader(
            sub_val_dataset,
            rank=args.rank,
            world_size=args.world_size,
            shuffle=False,
            drop_last=False,
            **cfg.solver.val,
        )    

    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    test_loader = build_dataloader(
        test_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )

    sub_test_dataset = build_dataset(cfg.dataset.sub_test, default_args=dict(logger=logger))
    sub_test_loader = build_dataloader(
        sub_test_dataset, 
        rank=args.rank, world_size=args.world_size, shuffle=False, drop_last=False, **cfg.solver.test,
    )

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")
        scaler = GradScaler()
    else:
        scaler = None

    # build model
    teacher_model = build_detector(deepcopy(cfg.teacher_model))
    student_model = build_detector(deepcopy(cfg.model))

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
        teacher_model.load_state_dict(t_checkpoint["state_dict"])

    # Load student param from teacher
    s_ckpt_path = cfg.model.backbone.custom['ckpt_path']
    logger.info(f"student model load ckpt from {s_ckpt_path}")
    s_checkpoint = torch.load(s_ckpt_path, map_location=device)
    student_model.load_state_dict(s_checkpoint['state_dict'], strict=False)
    if use_ema:
        student_model_ema.module.load_state_dict(s_checkpoint['state_dict_ema'], strict=False)

    # resume: reset epoch, load checkpoint / best rmse
    if args.resume != None:
        logger.info("Resume training from: {}".format(args.resume))
        device = f"cuda:{args.local_rank}"
        checkpoint = torch.load(args.resume, map_location=device)
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        student_model.load_state_dict(checkpoint['state_dict'])
        optimizer = build_optimizer(deepcopy(cfg.optimizer), student_model, logger)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler, max_epoch = build_scheduler(deepcopy(cfg.scheduler), optimizer, len(train_loader))
        scheduler.load_state_dict(checkpoint["scheduler"])
        if student_model_ema != None:
            student_model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

        del checkpoint  #  save memory if the model is very large such as ViT-g
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    # train the detector
    logger.info("Training Starts...\n")

    have_dropped_idx = cfg.model.backbone.backbone.drop_idx
    have_dropped_num = len(have_dropped_idx)
    tot_drop_num = cfg.model.backbone.backbone.depth // 4

    if args.rank == 0:
        print(f"have dropped {have_dropped_idx}")
        print(f"default drop {tot_drop_num} blocks")

    drop_type = getattr(cfg.solver, "drop_type", "train_mAP")
    if drop_type == "train_mAP" or drop_type == "block_mse":
        inference_loader = inference_train_loader
    elif drop_type == "train_loss":
        inference_loader = train_loader
    else:
        raise ValueError(f'Unsopport drop with {drop_type}')

    eval_drop_one_block_fun_dict = {"train_mAP": eval_drop_one_block_mAP, "train_loss": eval_drop_one_block_loss, "block_mse": eval_drop_one_block_mse}
    
    for drop_time in range(have_dropped_num + 1, tot_drop_num + 1):
        use_full_dataset = False
        if drop_time > tot_drop_num // 2:
            use_full_dataset = True

        next_drop_idx = eval_drop_one_block_fun_dict[drop_type](
                            inference_loader,
                            student_model,
                            cfg,
                            logger,
                            args.rank,
                            model_ema=student_model_ema,
                            use_amp=use_amp,
                            world_size=args.world_size,
                            not_eval=args.not_eval,
                        )

        # rebuild model
        if cfg.solver.dynamic_teacher:
            cfg.teacher_model.backbone.backbone.drop_idx = have_dropped_idx
        have_dropped_idx.append(next_drop_idx)
        cfg.model.backbone.backbone.drop_idx = have_dropped_idx
        if drop_time != have_dropped_num + 1:
            save_pth_path = os.path.join(cfg.work_dir, "checkpoint", "best.pth")
            t_ckpt_path = os.path.join(cfg.work_dir, "checkpoint", "merge.pth")
            print("merge model...", end='')
            MergeLoRA(save_pth_path, t_ckpt_path, cfg.model.backbone.backbone.depth)
            if getattr(cfg.solver, "is_merge", True):
                s_ckpt_path = t_ckpt_path
            else:
                save_pth_path = os.path.join(cfg.work_dir, "checkpoint", "best.pth")
                s_ckpt_path = os.path.join(cfg.work_dir, "checkpoint", f"best_{drop_time}.pth")
                print("copy model...", end='')
                shutil.copyfile(save_pth_path, s_ckpt_path)

        if cfg.solver.dynamic_teacher:
            teacher_model = build_detector(deepcopy(cfg.teacher_model))
        student_model = build_detector(deepcopy(cfg.model))

        # DDP
        use_static_graph = getattr(cfg.solver, "static_graph", False)
        if cfg.solver.dynamic_teacher:
            teacher_model = teacher_model.to(args.local_rank)
            teacher_model = DistributedDataParallel(teacher_model, device_ids=[args.local_rank], output_device=args.local_rank)

        student_model = student_model.to(args.local_rank)
        student_model = DistributedDataParallel(student_model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=False if use_static_graph else True,
            static_graph=use_static_graph,  # default is False, should be true when use activation checkpointing in E2E
        )

        # FP16 compression
        use_fp16_compress = getattr(cfg.solver, "fp16_compress", False)
        if use_fp16_compress:
            if cfg.solver.dynamic_teacher:
                teacher_model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
            student_model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

        # Model EMA
        use_ema = getattr(cfg.solver, "ema", False)
        if use_ema:
            student_model_ema = ModelEma(student_model)
        else:
            student_model_ema = None

        # Load Trained Model (Load student param from merge model)
        device = f"cuda:{args.local_rank}"
        t_checkpoint = torch.load(t_ckpt_path, map_location=device)
        s_checkpoint = torch.load(s_ckpt_path, map_location=device)
        logger.info(f"{drop_time}: student model load ckpt from {s_ckpt_path}")
        student_model.load_state_dict(s_checkpoint['state_dict'], strict=False)
        if use_ema:
            if cfg.solver.dynamic_teacher:
                teacher_model.module.load_state_dict(t_checkpoint["state_dict_ema"])
            student_model_ema.module.load_state_dict(s_checkpoint['state_dict_ema'], strict=False)
        else:
            if cfg.solver.dynamic_teacher:
                teacher_model.load_state_dict(t_checkpoint["state_dict"])
            student_model.load_state_dict(s_checkpoint['state_dict'], strict=False)

        # build optimizer and scheduler
        optimizer = build_optimizer(deepcopy(cfg.optimizer), student_model, logger)
        warmup_epoch = cfg.scheduler.warmup_epoch

        if use_full_dataset:
            scheduler, max_epoch = build_scheduler(deepcopy(cfg.scheduler), optimizer, len(train_loader))
            max_epoch = cfg.workflow.full_end_epoch
            val_first_start_epoch = cfg.workflow.full_val_first_start_epoch
            val_first_eval_interval = cfg.workflow.full_val_first_eval_interval
            val_second_start_epoch = cfg.workflow.full_val_second_start_epoch
            val_second_eval_interval = cfg.workflow.full_val_second_eval_interval
        else:
            scheduler, max_epoch = build_scheduler(deepcopy(cfg.sub_scheduler), optimizer, int(len(train_loader) * cfg.dataset.train.sample_ratio))
            max_epoch = cfg.workflow.sub_end_epoch
            val_first_start_epoch = cfg.workflow.sub_val_first_start_epoch
            val_first_eval_interval = cfg.workflow.sub_val_first_eval_interval
            val_second_start_epoch = cfg.workflow.sub_val_second_start_epoch
            val_second_eval_interval = cfg.workflow.sub_val_second_eval_interval

        best_mAP = 0.0
        val_loss_best = 1e6

        for epoch in range(resume_epoch + 1, max_epoch):
            # get sub dataset
            sub_train_loader = build_dataloader(
                Subset(train_dataset, get_sub_dataloader_idx(len(train_dataset), cfg.dataset.train.sample_ratio)), 
                rank=args.rank, world_size=args.world_size, shuffle=True, drop_last=True, **cfg.solver.train,
            )

            if use_full_dataset:
                train_loader.sampler.set_epoch(epoch)
            else:
                sub_train_loader.sampler.set_epoch(epoch)

            if cfg.cal_loss.use_dkd_loss:
                cfg.solver.weight.loss_cls_kd = min(epoch / warmup_epoch, 1.0)

            # train for one epoch
            train_one_epoch_dist(
                (train_loader if use_full_dataset else sub_train_loader),
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
                rank=args.rank,
            )

            # val for one epoch
            val_flag = False
            if (epoch + 1) >= val_first_start_epoch:
                if (cfg.workflow.val_loss_interval > 0) and ((epoch + 1) % cfg.workflow.val_loss_interval == 0):
                    val_flag = True

            if val_flag:
                val_loss = val_one_epoch(
                    (val_loader if use_full_dataset else sub_val_loader),
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
                        save_val_checkpoint(student_model, student_model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir)

            # eval for one epoch
            eval_flag = False
            if (epoch + 1) >= val_second_start_epoch:
                if val_second_eval_interval > 0 and (epoch + 1) % val_second_eval_interval == 0:
                    eval_flag = True
            elif (epoch + 1) >= val_first_start_epoch:
                if val_first_eval_interval > 0 and (epoch + 1) % val_first_eval_interval == 0:
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
                                    sub_set=False,
                                )

                if metrics_dict:
                    if type(test_loader.dataset).__name__[:6] == 'Thumos':
                        if best_mAP < metrics_dict["mAP@0.5"]:
                            best_mAP = metrics_dict["mAP@0.5"]
                            save_test_checkpoint(student_model, student_model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir)
                    else:
                        print("choose the best Average-mAP")
                        if best_mAP < metrics_dict['average_mAP']:
                            best_mAP = metrics_dict['average_mAP']
                            save_test_checkpoint(student_model, student_model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir)
        
        logger.info(f"done! best mAP is: {best_mAP}")

    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()