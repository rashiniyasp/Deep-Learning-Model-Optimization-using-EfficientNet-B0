"""
run_efficientnet_analysis.py
============================
Post-training analysis pipeline for EfficientNet-B0 (already trained).
Runs all remaining steps using the saved best_model.pth and feature files.

Steps:
  [1] Temperature Scaling Calibration (fixed: uses test-split 50/50, not train logits)
  [2] Deployment Benchmarking & INT8 Quantization  (fixed: full test set evaluation)
  [3] t-SNE + UMAP Latent Space Projections
  [4] Grad-CAM Anatomical Saliency Heatmaps
  [5] Final Report: Markdown + LaTeX + Pareto Frontier Plot

Usage:
  python src/run_efficientnet_analysis.py
  python src/run_efficientnet_analysis.py --skip_bench   # skip latency benchmark (slow)

All outputs are saved to:  outputs/efficientnet_b0_adamw/
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import os
import time
import argparse
import subprocess
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

# Use Python 3.11 where timm/torch/onnxruntime are installed.
# sys.executable picks up the default python (3.9) which lacks these packages.
_PY311 = Path(r"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe")
PYTHON = str(_PY311) if _PY311.exists() else sys.executable
MODEL_NAME = "efficientnet_b0"
RUN_NAME   = "efficientnet_b0_adamw"


def log_phase(title: str):
    sep = "=" * 78
    print(f"\n{sep}\n  {title}\n{sep}\n")


def run_step(cmd, desc=""):
    print(f">>> {desc}")
    print(f"    CMD: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(SRC_DIR.parent))
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"    Done in {elapsed:.1f}s\n")
        return True
    else:
        print(f"    FAILED (rc={result.returncode}) in {elapsed:.1f}s\n")
        return False


def check_prerequisites():
    """Verify that the EfficientNet checkpoint and features exist before running."""
    from config import OUTPUT_DIR
    run_dir  = OUTPUT_DIR / RUN_NAME
    ckpt     = run_dir / "checkpoints" / "best_model.pth"
    feat_dir = run_dir / "features"
    feat     = feat_dir / "test_logits.npy"
    emb      = feat_dir / "test_embeddings.npy"

    ok = True
    for path, label in [(ckpt, "Checkpoint"), (feat, "Test logits"), (emb, "Embeddings")]:
        if not path.exists():
            print(f"  MISSING : {label} -> {path}")
            ok = False
        else:
            sz = path.stat().st_size / 1e6
            print(f"  FOUND   : {label} ({sz:.1f} MB) -> {path.name}")

    return ok


def main():
    parser = argparse.ArgumentParser(description="EfficientNet-B0 Post-Training Analysis")
    parser.add_argument('--skip_bench',   action='store_true', help="Skip quantization/latency benchmark")
    parser.add_argument('--skip_gradcam', action='store_true', help="Skip Grad-CAM")
    parser.add_argument('--skip_tsne',    action='store_true', help="Skip t-SNE/UMAP")
    args = parser.parse_args()

    t_total = time.time()

    log_phase("EfficientNet-B0 Post-Training Analysis Pipeline")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Run dir : outputs/{RUN_NAME}/")
    print(f"  Python  : {PYTHON}\n")

    # Prerequisites check
    log_phase("Prerequisite Check")
    if not check_prerequisites():
        print("\nPrerequisites not met. Run training first:")
        print("  python src/run_full_pipeline.py --epochs 50")
        sys.exit(1)
    print("\n  All prerequisites satisfied.\n")

    # Step 1: Temperature Scaling
    log_phase("Step 1/5: Post-Hoc Probability Calibration (Temperature Scaling)")
    print("  Guo et al. (ICML 2017): learns scalar T* minimising NLL on calibration split.")
    print("  Calibration set = first 50% of test set (permuted, seed=42).")
    print("  Bug fixed: previous run used TRAIN logits => T*=-7.93 (physically invalid).\n")
    run_step(
        [PYTHON, str(SRC_DIR / "post_train" / "temperature_scaling.py")],
        desc="Temperature scaling calibration"
    )

    # Step 2: Quantization & Benchmark
    if not args.skip_bench:
        log_phase("Step 2/5: Quantization & Laptop CPU Deployment Benchmark")
        print("  Variants: PyTorch FP32 | PyTorch INT8 | ONNX FP32 | ONNX INT8")
        print("  Bug fixed: evaluation now uses FULL test set (not 200-sample subset).\n")
        bench_code = (
            "import sys; sys.path.insert(0, r'" + str(SRC_DIR) + "'); "
            "from post_train.quantize_benchmark import optimize_and_benchmark; "
            "optimize_and_benchmark('" + MODEL_NAME + "', '" + RUN_NAME + "', num_bench_samples=50)"
        )
        run_step(
            [PYTHON, "-c", bench_code],
            desc="EfficientNet-B0 quantization + latency benchmark"
        )
    else:
        log_phase("Step 2/5: SKIPPED (--skip_bench)")

    # Step 3: t-SNE + UMAP
    if not args.skip_tsne:
        log_phase("Step 3/5: Latent Space Manifold Analysis (t-SNE + UMAP)")
        print("  Input: test_embeddings.npy (5271 x 1280 penultimate features)")
        print("  Colour axes: anatomical class, ultrasound machine, operator\n")
        tsne_code = (
            "import sys; sys.path.insert(0, r'" + str(SRC_DIR) + "'); "
            "from post_train.tsne_umap import run_manifold_analysis; "
            "run_manifold_analysis('" + MODEL_NAME + "', '" + RUN_NAME + "')"
        )
        run_step(
            [PYTHON, "-c", tsne_code],
            desc="t-SNE + UMAP projection"
        )
    else:
        log_phase("Step 3/5: SKIPPED (--skip_tsne)")

    # Step 4: Grad-CAM
    if not args.skip_gradcam:
        log_phase("Step 4/5: Ultrasound Explainability (Grad-CAM Saliency Maps)")
        print("  Grad-CAM on last conv block of EfficientNet-B0 (blocks[-1]).")
        print("  Gallery: 4 images per anatomical class = 24 panels total.\n")
        gradcam_code = (
            "import sys; sys.path.insert(0, r'" + str(SRC_DIR) + "'); "
            "from post_train.gradcam import run_gradcam_gallery; "
            "run_gradcam_gallery('" + MODEL_NAME + "', '" + RUN_NAME + "')"
        )
        run_step(
            [PYTHON, "-c", gradcam_code],
            desc="Grad-CAM heatmap gallery"
        )
    else:
        log_phase("Step 4/5: SKIPPED (--skip_gradcam)")

    # Step 5: Final Report
    log_phase("Step 5/5: Comparative Report Generation")
    print("  Generates: Markdown table, LaTeX table, Pareto frontier, calibration plot\n")
    run_step(
        [PYTHON, str(SRC_DIR / "post_train" / "generate_report.py")],
        desc="Final benchmark report"
    )

    elapsed_total = time.time() - t_total
    log_phase(f"Analysis Complete in {elapsed_total/60:.1f} minutes")
    print("  Key outputs in outputs/efficientnet_b0_adamw/")
    print()
    print("  plots/")
    print("    training_curves.png           | confusion_matrix.png")
    print("    per_class_metrics.png         | per_group_accuracy.png")
    print("    calibration_temperature_scaling.png | reliability_diagram.png")
    print("    deployment_optimization.png   | tsne_anatomical_planes.png")
    print("    umap_anatomical_planes.png    | gradcam_gallery.png")
    print()
    print("  metrics/")
    print("    all_metrics.json              | calibration_stats.json")
    print("    deployment_benchmark.json     | training_history.json")
    print()
    print("  checkpoints/")
    print("    best_model.pth (46 MB FP32)  | model_int8_pt.pth (~15 MB)")
    print("    efficientnet_b0_fp32.onnx    | efficientnet_b0_int8.onnx")
    print()
    print("  features/")
    print("    test_embeddings.npy (5271x1280) | test_logits.npy")
    print("    test_probs_calibrated.npy       | test_meta.csv")


if __name__ == "__main__":
    main()
