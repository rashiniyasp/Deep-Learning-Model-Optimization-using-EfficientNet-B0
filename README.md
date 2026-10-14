# HTIC — Deep Learning Model Optimization & Edge Deployment
## Real-Time Fetal Ultrasound Plane Classification
**Healthcare Technology Innovation Centre (HTIC) & IGRS, IIT Madras**

---

## 🎯 Executive Summary

This repository contains the complete production-grade and research-oriented deep learning optimization pipeline for **real-time fetal ultrasound anatomical plane classification** deployed on an edge laptop CPU.

- **Model Architecture**: **EfficientNet-B0** (Compound Scaling CNN, 4.02M parameters, 0.39 GFLOPs)
- **Dataset**: `FETAL_PLANES_DB` (Zenodo 3904280 — 12,400 clinical ultrasound images across 6 anatomical planes)
- **Evaluation**: Strictly isolated test split (5,271 held-out images)
- **Deployment Hardware**: Laptop CPU without GPU (AMD Ryzen 7 7730U @ 2.00 GHz, 8 Cores / 16 Threads, 16GB RAM)
- **Baseline Accuracy**: **94.67% Test Accuracy**, **0.9406 Macro F1**, **0.9527 Macro PR-AUC**
- **Edge Deployment**: **ONNX Runtime (21.22 ms, 47.13 FPS)** & **PyTorch Dynamic INT8 (15.58 MB, 3.0× compression with 100% accuracy retention)**
- **Medical Reliability**: Post-hoc **Temperature Scaling Calibration** ($T^* = 0.8765$, reducing ECE by 28.8% from 13.27% to 9.46%)
- **Clinical Interpretability**: Penultimate **1,280-dim t-SNE & UMAP** latent manifold analysis and **Grad-CAM** anatomical acoustic landmark verification

---

## 🚀 One-Click Master Pipeline Execution

To run the complete post-training analysis pipeline (benchmarking, probability calibration, manifold projections, Grad-CAM, and report synthesis):

```bash
cd r:\HTIC
python src/run_efficientnet_analysis.py
```

### Other Run Commands:
- **Download Pre-trained Checkpoints & Embeddings**:
  ```bash
  python src/download_weights.py
  ```
- **Full End-to-End Pipeline (Training + Post-Training)**:
  ```bash
  python src/run_full_pipeline.py
  ```
- **Post-Training Analysis Only (Skip training, evaluate checkpoints)**:
  ```bash
  python src/run_full_pipeline.py --skip_train
  ```

---

## 📊 Empirical Results Summary

### 1. Classification & Medical Reliability Metrics
*Evaluated on all 5,271 held-out test images:*

| Metric | Empirical Value |
|---|---|
| **Test Accuracy** | **94.67%** (4,990 / 5,271 correct) |
| **Macro F1-Score** | **0.9406** |
| **Weighted F1-Score** | **0.9467** |
| **Macro PR-AUC** | **0.9527** |
| **Uncalibrated ECE** | 13.27% |
| **Calibrated ECE** | **9.46%** (**28.8% relative reduction**) |
| **Optimal Temperature ($T^*$)** | **0.8765** |

### 2. Laptop CPU Deployment Benchmarks (AMD Ryzen 7 7730U)
*50 measured iterations per deployment variant:*

| Deployment Variant | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) | Throughput (FPS) | Test Accuracy (%) | Model Size (MB) | Compression |
|---|---|---|---|---|---|---|---|
| **PyTorch FP32 (Baseline)** | 21.34 ms | 21.62 ms | 22.41 ms | 46.87 FPS | **94.67%** | 46.40 MB | 1.0× (Baseline) |
| **PyTorch INT8 (Dynamic)** | 22.15 ms | 22.21 ms | 23.48 ms | 45.15 FPS | **94.67%** | **15.58 MB** | **3.0× smaller** |
| **ONNX FP32 (Graph Fused)** | **21.22 ms** | **22.05 ms** | **24.33 ms** | **47.13 FPS** | **94.67%** | **15.30 MB** | **3.0× smaller** |
| **ONNX Dynamic INT8** | 135.84 ms | 137.17 ms | 151.59 ms | 7.36 FPS | 29.67% | **4.14 MB** | **11.2× smaller** |

---

## 📋 Implemented Pipeline Overview

| Phase | Module | Implemented Methodology & Relevance | Key Output Artifacts |
|---|---|---|---|
| **1. Training & Convergence** | `src/train.py` | Fine-tuned ImageNet-21k weights with AdamW, Cosine Annealing, Label Smoothing 0.1, and early stopping at Epoch 24 | `checkpoints/best_model.pth`<br>`metrics/all_metrics.json`<br>`plots/training_curves.png` |
| **2. Feature Extraction** | `src/feature_extractor.py` | Extracts penultimate 1,280-dim feature vectors, logits, probabilities, and clinical metadata for train/test splits | `features/train_embeddings.npy`<br>`features/test_embeddings.npy`<br>`features/test_meta.csv` |
| **3. Deployment Benchmarks** | `src/post_train/quantize_benchmark.py` | PyTorch dynamic INT8 quantization + ONNX graph export + ONNX Runtime CPU latency, FPS, disk footprint, and accuracy benchmarking | `checkpoints/*.onnx`<br>`metrics/deployment_benchmark.json`<br>`plots/deployment_optimization.png` |
| **4. Confidence Calibration** | `src/post_train/temperature_scaling.py` | Post-hoc Temperature Scaling (Guo et al., ICML 2017) using L-BFGS with positive reparameterization $T = \exp(\log T) > 0$ to reduce ECE | `features/test_probs_calibrated.npy`<br>`metrics/calibration_stats.json`<br>`plots/calibration_temperature_scaling.png` |
| **5. Manifold Visualization** | `src/post_train/tsne_umap.py` | High-dimensional latent projections via t-SNE and UMAP, analyzing anatomical plane separability, operator consistency, and scanner invariance | `features/2d_projections.npz`<br>`plots/tsne_anatomical_planes.png`<br>`plots/umap_anatomical_planes.png` |
| **6. Explainability** | `src/post_train/gradcam.py` | Native PyTorch Grad-CAM on final MBConv block (`blocks[-1]`), overlaying heatmaps on original ultrasound frames across all 6 planes | `plots/gradcam_gallery.png` |
| **7. Report Synthesis** | `src/post_train/generate_report.py` | Synthesizes Markdown & LaTeX tables, Latency vs Accuracy Pareto frontier plot, and calibration bar charts | `outputs/final_benchmark_table.md`<br>`outputs/final_benchmark_table.tex`<br>`outputs/pareto_frontier_latency_f1.png`<br>`outputs/calibration_summary.png` |

---

## 🗂️ Project Directory Structure

```
HTIC/
├── requirements.txt
├── README.md
├── ASSIGNMENT_PLAN.md               ← Complete implemented project strategy
├── ASSIGNMENT_TECHNICAL_NOTES.md   ← In-depth methodology, math & interview discussion
├── EXPERIMENT_GUIDE.md             ← Theoretical handbook & clinical problem formulation
├── src/
│   ├── config.py                   ← Central hyperparameters, paths, and model specs
│   ├── dataset.py                  ← Data loading, transforms, and stratified split
│   ├── models.py                   ← Model builder with get_features() embedding hook
│   ├── metrics.py                  ← Accuracy, F1, PR-AUC, ECE, and subgroup auditing
│   ├── feature_extractor.py        ← Penultimate embedding & prediction extractor
│   ├── train.py                    ← EfficientNet-B0 training loop
│   ├── run_full_pipeline.py        ← Master pipeline orchestrator
│   ├── run_efficientnet_analysis.py← Fast post-training benchmark & analysis runner
│   └── post_train/
│       ├── quantize_benchmark.py   ← PyTorch INT8, ONNX export, and CPU benchmarking
│       ├── temperature_scaling.py  ← L-BFGS temperature scaling calibration
│       ├── tsne_umap.py            ← t-SNE & UMAP manifold projections
│       ├── gradcam.py              ← Ultrasound Grad-CAM explainability
│       └── generate_report.py      ← Summary reports & Pareto frontier generator
└── outputs/
    ├── final_benchmark_table.md    ← Markdown benchmark table
    ├── final_benchmark_table.tex   ← LaTeX table for paper / presentation
    ├── pareto_frontier_latency_f1.png ← Latency vs Accuracy Pareto frontier
    ├── calibration_summary.png     ← Before vs After calibration ECE comparison
    └── efficientnet_b0_adamw/
        ├── checkpoints/            (best_model.pth, model_int8_pt.pth, ONNX models)
        ├── features/               (embeddings.npy, logits.npy, probs.npy, meta.csv)
        ├── metrics/                (all_metrics.json, deployment_benchmark.json, ...)
        └── plots/                  (training curves, confusion matrix, t-SNE, UMAP, Grad-CAM)
```
