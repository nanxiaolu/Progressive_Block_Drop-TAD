import os
import copy
import json
import tqdm
import torch
import torch.distributed as dist

from opentad.utils import create_folder, search_drop_idx
from opentad.models.utils.post_processing import build_classifier, batched_nms
from opentad.evaluations import build_evaluator
from opentad.datasets.base import SlidingWindowDataset

def gather_ddp_results(world_size, result_dict, post_cfg):
    gather_dict_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_dict_list, result_dict)
    result_dict = {}
    for i in range(world_size):  # update the result dict
        for k, v in gather_dict_list[i].items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    # do nms for sliding window, if needed
    if post_cfg.sliding_window == True and post_cfg.nms is not None:
        # assert sliding_window=True
        tmp_result_dict = {}
        for k, v in result_dict.items():
            segments = torch.Tensor([data["segment"] for data in v])
            scores = torch.Tensor([data["score"] for data in v])
            labels = []
            class_idx = []
            for data in v:
                if data["label"] not in class_idx:
                    class_idx.append(data["label"])
                labels.append(class_idx.index(data["label"]))
            labels = torch.Tensor(labels)

            segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=class_idx[int(label.item())],
                        score=round(score.item(), 4),
                    )
                )
            tmp_result_dict[k] = results_per_video
        result_dict = tmp_result_dict
    return result_dict

def eval_drop_one_block(
    inference_train_loader,
    model,
    cfg,
    logger,
    rank,
    model_ema=None,
    use_amp=False,
    world_size=0,
    not_eval=False,
):
    """Inference and Evaluation the model"""
    drop_list = search_drop_idx(model.state_dict(), cfg.model.backbone.backbone.depth)
    # drop 4 -> [[1], [2], [3], [5], [6], [7], [8], [9], [10], [11]]
    cur_drop_list = model.module.backbone.model.backbone.drop_idx
    result_dict = [{} for _ in range(len(drop_list))]
    metrics_dict = [{} for _ in range(len(drop_list))]

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.state_dict())

    cfg.inference["folder"] = os.path.join(cfg.work_dir, "outputs")
    if cfg.inference.save_raw_prediction:
        create_folder(cfg.inference["folder"])

    # external classifier
    if "external_cls" in cfg.post_processing:
        if cfg.post_processing.external_cls != None:
            external_cls = build_classifier(cfg.post_processing.external_cls)
    else:
        external_cls = inference_train_loader.dataset.class_map

    # whether the testing dataset is sliding window
    cfg.post_processing.sliding_window = isinstance(inference_train_loader.dataset, SlidingWindowDataset)

    # model forward
    model.eval()

    for data_dict in tqdm.tqdm(inference_train_loader, disable=(rank != 0)):
        torch.distributed.barrier()
        for idx, drop_idx in enumerate(drop_list):
            model.module.backbone.model.backbone.drop_idx = cur_drop_list + drop_idx
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                with torch.no_grad():
                    results = model(
                        **data_dict,
                        return_loss=False,
                        infer_cfg=cfg.inference,
                        post_cfg=cfg.post_processing,
                        ext_cls=external_cls,
                    )

            # update the result dict
            for k, v in results.items():
                if k in result_dict[idx].keys():
                    result_dict[idx][k].extend(v)
                else:
                    result_dict[idx][k] = v

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)

    # process result
    for idx in range(len(result_dict)):
        result_dict[idx] = gather_ddp_results(world_size, result_dict[idx], cfg.post_processing)

        result_eval = dict(results=result_dict[idx])
        if cfg.post_processing.save_dict:
            result_path = os.path.join(cfg.work_dir, "result_detection.json")
            with open(result_path, "w") as out:
                json.dump(result_eval, out)

        if not not_eval:
            # build evaluator
            evaluator = build_evaluator(dict(prediction_filename=result_eval, **cfg.train_evaluation))
            # evaluate and output
            logger.info(f"\n\n{drop_list[idx][0]} block evaluation starts...")
            metrics_dict[idx] = evaluator.evaluate()
            evaluator.logging(logger)

    best_mAP = 0.
    for idx in range(len(metrics_dict)):
        if type(inference_train_loader.dataset).__name__[:6] == 'Thumos':
            if best_mAP < metrics_dict[idx]["mAP@0.5"]:
                best_mAP = metrics_dict[idx]["mAP@0.5"]
                best_idx = idx
        else:
            print("choose the best Average-mAP")
            if best_mAP < metrics_dict[idx]['average_mAP']:
                best_mAP = metrics_dict[idx]['average_mAP']
                best_idx = idx

    return drop_list[best_idx][0]
