import copy
import torch
import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss
global_step = 0

def train_one_epoch_dist(
    train_loader,
    teacher_model,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    logger,
    model_ema=None,
    clip_grad_l2norm=-1,
    logging_interval=200,
    scaler=None,
    train_writer=None,
    weight=None,
    rank=0,
):
    """Training the model for one epoch"""
    global global_step
    logger.info("[Train]: Epoch {:d} started".format(curr_epoch))
    losses_tracker = {}
    use_amp = False if scaler is None else True

    teacher_model.eval()
    model.train()

    for data_dict in tqdm.tqdm(train_loader, disable=(rank != 0)):
        torch.distributed.barrier()
        global_step += 1
        optimizer.zero_grad()

        # forward pass
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                feat_mid_list, feat_out_list, cls_target, reg_target = teacher_model(**data_dict, return_loss=True, is_teacher=True)

        data_dict['feat_mid_list'] = feat_mid_list
        data_dict['feat_out_list'] = feat_out_list

        if cls_target != None:
            data_dict['cls_target'] = cls_target

        if reg_target != None:
            data_dict['reg_target'] = reg_target

        data_dict['is_student'] = True

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            losses = model(**data_dict, return_loss=True)
            losses["cost"] = sum(_value * weight[_key] for _key, _value in losses.items())
        # compute the gradients
        if use_amp:
            scaler.scale(losses["cost"]).backward()
        else:
            losses["cost"].backward()

        # gradient clipping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)

        # update parameters
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # update scheduler
        scheduler.step()

        # update ema
        if model_ema is not None:
            model_ema.update(model)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        # tensorboard
        if train_writer:
            for key, value in losses.items():
                train_writer.add_scalar(key, value.item(), global_step)
            train_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)


def val_one_epoch(
    val_loader,
    model,
    logger,
    rank,
    curr_epoch,
    model_ema=None,
    use_amp=False,
):
    """Validating the model for one epoch: compute the loss"""

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.state_dict())

    logger.info("[Val]: Epoch {:d} Loss".format(curr_epoch))
    losses_tracker = {}

    model.eval()
    for data_dict in tqdm.tqdm(val_loader, disable=(rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                losses = model(**data_dict, return_loss=True)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

    # print to terminal
    block1 = "[Val]: [{:03d}]".format(curr_epoch)
    block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
    block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
    logger.info("  ".join([block1, block2, "  ".join(block3)]))

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)
    return losses_tracker["cost"].avg
