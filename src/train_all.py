"""
train_all.py — Master script: runs all 3 models sequentially overnight.

Usage:
    python src/train_all.py             # full 50-epoch run
    python src/train_all.py --test_run  # 2-epoch smoke test

Models trained (in order):
  1. efficientnet_b0      (AdamW, no CutMix) — baseline
  2. repvit_m1            (ASAM + CutMix)    — primary
  3. mobileone_s2         (ASAM + CutMix)    — secondary comparison
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import subprocess
import argparse
import time
from pathlib import Path

PYTHON = sys.executable   # same python that launched this script

MODELS = [
    {
        'name'      : 'efficientnet_b0',
        'extra_args': [],
        'description': '① EfficientNet-B0  (AdamW baseline)',
    },
    {
        'name'      : 'repvit_m1',
        'extra_args': ['--use_asam', '--use_cutmix'],
        'description': '② RepViT-M1.0      (ASAM + CutMix)',
    },
    {
        'name'      : 'mobileone_s2',
        'extra_args': ['--use_asam', '--use_cutmix'],
        'description': '③ MobileOne-S2     (ASAM + CutMix)',
    },
]


def run_model(model_cfg: dict, epochs: int, test_run: bool):
    script = str(Path(__file__).parent / "train.py")
    cmd = [
        PYTHON, script,
        '--model',  model_cfg['name'],
        '--epochs', str(epochs),
    ] + model_cfg['extra_args']

    if test_run:
        cmd.append('--test_run')

    print("\n" + "=" * 70)
    print(f"  {model_cfg['description']}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 70 + "\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n✅ {model_cfg['name']} finished in {elapsed/60:.1f} min")
    else:
        print(f"\n❌ {model_cfg['name']} FAILED (exit code {result.returncode})")
        print("   Check logs above. Continuing to next model...")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_run', action='store_true',
                        help='Run smoke test (2 epochs each)')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    epochs   = 2 if args.test_run else args.epochs
    results  = {}
    t_total  = time.time()

    print("\n🌙 OVERNIGHT TRAINING — FETAL PLANES CLASSIFICATION")
    print(f"   Models : {[m['name'] for m in MODELS]}")
    print(f"   Epochs : {epochs} {'(TEST RUN)' if args.test_run else ''}\n")

    for model_cfg in MODELS:
        success = run_model(model_cfg, epochs, args.test_run)
        results[model_cfg['name']] = '✅' if success else '❌'

    total_time = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"  ALL MODELS DONE  |  Total time: {total_time/3600:.1f} h")
    print("  Results:")
    for name, status in results.items():
        print(f"    {status}  {name}")
    print("=" * 70)
    print("  Output saved in: r:\\HTIC\\outputs\\")
    print("  Each model has: checkpoints/, features/, metrics/, plots/")


if __name__ == "__main__":
    main()
