"""
run_full_pipeline.py — Master One-Click Pipeline for Fetal Ultrasound Optimization & Deployment.

Runs the complete EfficientNet-B0 pipeline end-to-end:
  [Phase 1] Training & Feature Extraction:
            - EfficientNet-B0 (AdamW Baseline with Label Smoothing 0.1)
            - Saves best_model.pth, last_epoch.pth
            - Saves train & test penultimate embeddings (1280-dim), logits, probs, preds, metadata
            - Computes Accuracy, Macro/Weighted F1, PR-AUC, ECE, Subgroup Robustness
  [Phase 2] Laptop CPU Deployment Benchmarking & INT8 Quantization:
            - PyTorch FP32 Baseline
            - PyTorch Dynamic INT8 Quantization
            - ONNX Runtime FP32 Graph Export
            - ONNX Dynamic INT8 Quantization
  [Phase 3] Post-Hoc Confidence Calibration:
            - Temperature Scaling with L-BFGS on test-split calibration fold
            - Minimizes Negative Log-Likelihood & Expected Calibration Error
  [Phase 4] Latent Space Manifold Analysis:
            - t-SNE & UMAP 2D projections of 1280-dim feature space
  [Phase 5] Ultrasound Clinical Explainability:
            - Grad-CAM anatomical saliency heatmaps on penultimate MBConv block
  [Phase 6] Final Synthesis:
            - Markdown & LaTeX comparative benchmark tables
            - Latency vs Accuracy Pareto frontier chart
            - Reliability & calibration summary diagrams

Usage:
  python src/run_full_pipeline.py              # Full training + complete analysis
  python src/run_full_pipeline.py --test_run   # Quick 2-epoch sanity check of full pipeline
  python src/run_full_pipeline.py --skip_train # Post-training analysis on existing checkpoints
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
from config import OUTPUT_DIR, EPOCHS, MODELS

_PY311 = Path(r"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe")
PYTHON = str(_PY311) if _PY311.exists() else sys.executable


def log_phase(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78 + "\n")


def run_command(cmd, desc="Running step"):
    print(f"▶ {desc}...")
    print(f"  Command: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(SRC_DIR.parent))
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✅ Completed in {elapsed:.1f}s\n")
        return True
    else:
        print(f"  ❌ FAILED with return code {result.returncode} ({elapsed:.1f}s)\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Master Pipeline: Train, Optimize & Deploy EfficientNet-B0")
    parser.add_argument('--epochs',     type=int, default=EPOCHS, help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument('--test_run',   action='store_true',     help="Run 2-epoch smoke test across all steps")
    parser.add_argument('--skip_train', action='store_true',     help="Skip model training and run post-training analysis")
    args = parser.parse_args()

    total_start = time.time()
    n_epochs = 2 if args.test_run else args.epochs

    log_phase(f"🚀 STARTING FULL PIPELINE {'[TEST RUN]' if args.test_run else f'[{n_epochs} EPOCHS]'}")
    print(f"  Python executable: {PYTHON}")
    print(f"  Root directory   : {SRC_DIR.parent}")
    print(f"  Outputs directory: {OUTPUT_DIR}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Model Training & Feature Extraction
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_train:
        log_phase("PHASE 1: Model Training & Feature Extraction (EfficientNet-B0)")
        for model_cfg in MODELS:
            m_name = model_cfg['name']
            use_asam = model_cfg.get('use_asam', False)
            use_cutmix = model_cfg.get('use_cutmix', False)

            cmd = [
                PYTHON, str(SRC_DIR / "train.py"),
                '--model', m_name,
                '--epochs', str(n_epochs),
                '--no_wandb',
            ]
            if use_asam:
                cmd.append('--use_asam')
            if use_cutmix:
                cmd.append('--use_cutmix')
            if args.test_run:
                cmd.append('--test_run')

            success = run_command(cmd, desc=f"Training {m_name} (AdamW Baseline)")
            if not success:
                print(f"⚠️ Training {m_name} encountered an error. Proceeding to next step...")
    else:
        log_phase("PHASE 1: Skipping Training (--skip_train specified)")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: Deployment Benchmarking & INT8 Quantization
    # ─────────────────────────────────────────────────────────────────────────
    log_phase("PHASE 2: Laptop CPU Deployment Benchmarking & INT8 Quantization")
    cmd_bench = [PYTHON, str(SRC_DIR / "post_train" / "quantize_benchmark.py")]
    run_command(cmd_bench, desc="Benchmarking PyTorch FP32, INT8, and ONNX Runtime CPU latency")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3: Temperature Scaling (Clinical Confidence Calibration)
    # ─────────────────────────────────────────────────────────────────────────
    log_phase("PHASE 3: Post-Hoc Probability Calibration (Temperature Scaling)")
    cmd_calib = [PYTHON, str(SRC_DIR / "post_train" / "temperature_scaling.py")]
    run_command(cmd_calib, desc="Optimizing temperature T with L-BFGS to minimize Expected Calibration Error")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4: Latent Space Manifold Analysis (t-SNE & UMAP)
    # ─────────────────────────────────────────────────────────────────────────
    log_phase("PHASE 4: High-Dimensional Latent Manifold Projections (t-SNE & UMAP)")
    cmd_tsne = [PYTHON, str(SRC_DIR / "post_train" / "tsne_umap.py")]
    run_command(cmd_tsne, desc="Generating t-SNE & UMAP plots across classes, operators, and ultrasound devices")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 5: Explainability & Grad-CAM Visualizations
    # ─────────────────────────────────────────────────────────────────────────
    log_phase("PHASE 5: Ultrasound Clinical Explainability (Grad-CAM)")
    cmd_gradcam = [PYTHON, str(SRC_DIR / "post_train" / "gradcam.py")]
    run_command(cmd_gradcam, desc="Generating Grad-CAM anatomical saliency heatmaps for all fetal planes")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 6: Final Report Generation & Pareto Frontier Plot
    # ─────────────────────────────────────────────────────────────────────────
    log_phase("PHASE 6: Aggregating Results & Generating Comparative Reports")
    cmd_report = [PYTHON, str(SRC_DIR / "post_train" / "generate_report.py")]
    run_command(cmd_report, desc="Synthesizing Markdown table, LaTeX table, and Pareto frontier chart")

    total_time = time.time() - total_start
    log_phase(f"🎉 FULL PIPELINE COMPLETED in {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print("  All artifacts, weights, features, metrics, tables, and figures are saved:")
    print("  📁 outputs/")
    print("     ├── final_benchmark_table.md         (Markdown summary table)")
    print("     ├── final_benchmark_table.tex        (LaTeX table for paper)")
    print("     ├── pareto_frontier_latency_f1.png   (Pareto frontier: Latency vs Accuracy)")
    print("     ├── calibration_summary.png          (ECE Before vs After Temperature Scaling)")
    print("     └── efficientnet_b0_adamw/")
    print("         ├── checkpoints/ (best_model.pth, model_int8_pt.pth, ONNX models)")
    print("         ├── features/    (embeddings.npy, logits.npy, probs.npy, preds.npy, meta.csv)")
    print("         ├── metrics/     (all_metrics.json, deployment_benchmark.json, calibration_stats.json)")
    print("         └── plots/       (training curves, confusion matrix, t-SNE, UMAP, Grad-CAM)")


if __name__ == "__main__":
    main()
