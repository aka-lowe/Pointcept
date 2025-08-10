# tools/online_tta.py

import argparse
import os
import torch
import torch.optim as optim
from torch.nn import functional as F
from copy import deepcopy
import numpy as np
from collections import defaultdict

# Pointcept-specific imports
from pointcept.models import build_model
from pointcept.datasets import build_dataset, build_dataloader, collate_fn
from pointcept.utils.config import Config, DictAction
from pointcept.utils.logger import get_root_logger
from pointcept.utils.misc import (
    setup_environment,
    collect_env,
    build_model_from_cfg,
    to_dict,
)

# Import evaluator for metrics
from pointcept.utils.eval import SemSegEvaluator


def get_parser():
    """Parses command line arguments for the online TTA script."""
    parser = argparse.ArgumentParser(description="Pointcept Online TTA")
    parser.add_argument("config", help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--weight", type=str, default=None, help="path to pre-trained model weight")
    parser.add_argument("--options", nargs="+", action=DictAction, help="override some settings in the used config")
    return parser


def adapt_model_tent(cfg, model, batch):
    """Adapts the model on a single batch of data using the Tent strategy."""
    params_to_update = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    if not params_to_update:
        params_to_update = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]
    if not params_to_update: return

    optimizer = optim.Adam(params_to_update, lr=cfg.tta.lr)
    model.train()
    for _ in range(cfg.tta.adaptation_steps):
        output_dict = model(batch)
        logits = output_dict["logits"]
        entropy = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        optimizer.zero_grad()
        entropy.backward()
        optimizer.step()


def evaluate_scene(model, scene_loader, num_classes, ignore_index):
    """
    Evaluates the model's performance on all data from a single scene.
    """
    evaluator = SemSegEvaluator(num_classes=num_classes, ignore_index=ignore_index)
    model.eval()

    with torch.no_grad():
        for batch in scene_loader:
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()
            
            result = model(batch)
            evaluator.update(result["predict"].cpu().numpy(), batch["segment"].cpu().numpy())
    
    return evaluator.evaluate()


def run_online_tta(cfg, model, dataset, initial_state):
    """
    The main loop for running online TTA with scene-based reset and evaluation.
    """
    logger = get_root_logger()
    logger.info(">>> Starting Online Test-Time Adaptation with Scene-by-Scene Evaluation...")

    scene_data_indices = defaultdict(list)
    for i in range(len(dataset)):
        scene_name = dataset.get_data_name(i)
        scene_data_indices[scene_name].append(i)
    
    scene_names = sorted(scene_data_indices.keys())
    logger.info(f"Found {len(scene_names)} scenes to process.")
    
    # <<<====== MODIFICATION 1: Create lists to store all metrics ======>>>
    all_miou, all_macc, all_acc = [], [], []

    for scene_name in scene_names:
        logger.info(f"===== Processing scene: {scene_name} =====")
        
        logger.info("Resetting model to its initial pre-trained state.")
        model.load_state_dict(initial_state)

        scene_indices = scene_data_indices[scene_name]
        scene_subset = torch.utils.data.Subset(dataset, scene_indices)
        scene_loader = build_dataloader(
            scene_subset, batch_size=cfg.batch_size, num_workers=cfg.num_worker,
            collate_fn=collate_fn, sampler=None, shuffle=False
        )

        for i, batch in enumerate(scene_loader):
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()
            adapt_model_tent(cfg, model, batch)
        logger.info(f"Finished adapting on {len(scene_loader)} batches for scene '{scene_name}'.")

        logger.info(f"Evaluating performance on scene '{scene_name}'...")
        scene_metrics = evaluate_scene(
            model=model, scene_loader=scene_loader,
            num_classes=cfg.model.num_classes,
            ignore_index=cfg.data_stream.get("ignore_index", -1)
        )
        
        # <<<====== MODIFICATION 2: Extract and log all metrics ======>>>
        miou = scene_metrics['miou']
        macc = scene_metrics['macc']
        allacc = scene_metrics['allacc']

        logger.info(f"Scene '{scene_name}' Result: mIoU={miou:.2f}, mAcc={macc:.2f}, allAcc={allacc:.2f}")
        all_miou.append(miou)
        all_macc.append(macc)
        all_acc.append(allacc)

    # <<<====== MODIFICATION 3: Log the final average of all metrics ======>>>
    if all_miou:
        avg_miou = sum(all_miou) / len(all_miou)
        avg_macc = sum(all_macc) / len(all_macc)
        avg_allacc = sum(all_acc) / len(all_acc)
        logger.info("=" * 60)
        logger.info(">>> Online TTA Finished <<<")
        logger.info(f"Average performance across all {len(all_miou)} scenes:")
        logger.info(f"  - Average mIoU:   {avg_miou:.2f}%")
        logger.info(f"  - Average mAcc:   {avg_macc:.2f}%")
        logger.info(f"  - Average allAcc: {avg_allacc:.2f}%")
        logger.info("=" * 60)


def main():
    args = get_parser().parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    setup_environment(cfg)
    logger = get_root_logger()
    logger.info(f"Config:\n{cfg.pretty_text}")

    model = build_model_from_cfg(cfg.model).cuda()
    
    if args.weight:
        checkpoint = torch.load(args.weight, map_location="cpu")
        model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
        logger.info(f"Successfully loaded initial weights from: {args.weight}")
    else:
        logger.warning("TTA requires a pre-trained model weight.")
        return

    initial_model_state = deepcopy(model.state_dict())
    logger.info("Stored initial model state for scene resets.")

    if not hasattr(cfg, 'data_stream'):
        raise ValueError("Config must contain a 'data_stream' dictionary.")
    
    dataset = build_dataset(cfg.data_stream)

    run_online_tta(cfg, model, dataset, initial_model_state)


if __name__ == "__main__":
    main()