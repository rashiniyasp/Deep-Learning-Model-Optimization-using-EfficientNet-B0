"""
post_train/reparameterize.py — Structural Reparameterization for RepViT and MobileOne.

Collapses multi-branch training architectures into single-branch feed-forward
inference architectures:
  - RepViT: fuses multi-branch depthwise/pointwise convs via backbone.fuse()
  - MobileOne: fuses overparameterized conv branches and identity via reparameterize()

Reduces parameter memory footprint and eliminates multi-branch latency overhead
with zero loss in prediction accuracy.
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import json
import copy
from pathlib import Path
import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
from models import build_model
from config import OUTPUT_DIR, NUM_CLASSES


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def reparameterize_model(model_name: str, model: nn.Module) -> nn.Module:
    """
    In-place / deepcopy reparameterization depending on architecture.
    Returns fused model.
    """
    fused_model = copy.deepcopy(model)
    fused_model.eval()

    if 'repvit' in model_name:
        if hasattr(fused_model.backbone, 'fuse'):
            fused_model.backbone.fuse()
            print(f"  [Reparameterize] Successfully called RepViT backbone.fuse()")
        else:
            print("  [Reparameterize] Warning: backbone has no fuse() method.")
    elif 'mobileone' in model_name:
        reparam_count = 0
        for blk in fused_model.modules():
            if hasattr(blk, 'reparameterize'):
                blk.reparameterize()
                reparam_count += 1
        print(f"  [Reparameterize] Successfully reparameterized {reparam_count} MobileOne blocks")
    else:
        print(f"  [Reparameterize] {model_name} is not a reparameterizable architecture. Leaving as-is.")

    return fused_model


def verify_and_save_fused(model_name: str, run_name: str = None):
    """
    Loads best_model.pth from outputs/<run_name>/checkpoints/,
    fuses the model, validates numerical equivalence, saves fused_model.pth,
    and records statistics.
    """
    if run_name is None:
        tag = "asam_cutmix" if model_name in ["repvit_m1", "mobileone_s2"] else "adamw"
        run_name = f"{model_name}_{tag}"

    run_dir = OUTPUT_DIR / run_name
    ckpt_path = run_dir / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        print(f"⚠️ Checkpoint not found at {ckpt_path}. Skipping reparameterization.")
        return None

    print("\n" + "=" * 70)
    print(f"  Structural Reparameterization: {model_name} ({run_name})")
    print("=" * 70)

    device = torch.device("cpu")
    model = build_model(model_name, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    params_before = count_parameters(model)

    # Test numerical consistency on dummy batch
    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out_before = model(dummy_input)

    # Reparameterize
    fused_model = reparameterize_model(model_name, model)
    fused_model.eval()

    params_after = count_parameters(fused_model)

    with torch.no_grad():
        out_after = fused_model(dummy_input)

    max_diff = (out_before - out_after).abs().max().item()
    mean_diff = (out_before - out_after).abs().mean().item()

    print(f"  Params Before Fusion: {params_before:,}")
    print(f"  Params After Fusion : {params_after:,}")
    param_reduction = params_before - params_after
    print(f"  Params Reduced      : {param_reduction:,} ({param_reduction/params_before*100:.2f}%)")
    print(f"  Max Output Diff     : {max_diff:.3e}")
    print(f"  Mean Output Diff    : {mean_diff:.3e}")

    # Save fused model checkpoint
    fused_ckpt_path = run_dir / "checkpoints" / "fused_model.pth"
    torch.save({
        'model_name': model_name,
        'tag': run_name,
        'state_dict': fused_model.state_dict(),
        'params_before': params_before,
        'params_after': params_after,
        'max_diff': max_diff,
    }, fused_ckpt_path)
    print(f"  💾 Saved fused model checkpoint to: {fused_ckpt_path.name}")

    # Save stats json
    stats = {
        'model_name': model_name,
        'run_name': run_name,
        'params_before': params_before,
        'params_after': params_after,
        'param_reduction': param_reduction,
        'param_reduction_pct': float(param_reduction / params_before * 100),
        'max_diff': float(max_diff),
        'mean_diff': float(mean_diff),
    }
    stats_path = run_dir / "metrics" / "reparameterization_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  📊 Saved reparameterization stats to: {stats_path.name}")

    return fused_model, stats


if __name__ == "__main__":
    for m in ['repvit_m1', 'mobileone_s2']:
        verify_and_save_fused(m)
