# Comprehensive Model Optimization & Deployment Benchmark Report

**Target Architecture**: EfficientNet-B0 (Trained with AdamW on FETAL_PLANES_DB)
**Deployment Hardware**: Edge Laptop CPU (AMD Ryzen 7 7730U @ 2.00 GHz, 8 Cores / 16 Threads)
**Evaluation Dataset**: 5,271 strictly isolated test images

## 1. Classification & Medical Reliability Metrics

| Model           | Tag   | Accuracy (%) | Macro F1 | Weighted F1 | Macro PR-AUC | ECE Before Calib (%) | ECE After Calib (%) | Optimal Temp (T) | ECE Reduction (%) |
| --------------- | ----- | ------------ | -------- | ----------- | ------------ | -------------------- | ------------------- | ---------------- | ----------------- |
| efficientnet_b0 | adamw | 94.6689      | 0.9406   | 0.9467      | 0.9527       | 13.2738              | 9.4553              | 0.8765           | 28.7668           |

## 2. Laptop Deployment Optimization Benchmark

| Deployment Variant      | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) | Throughput (FPS) | Test Accuracy (%) | Model Size (MB) | Compression |
| ----------------------- | ----------------- | ---------------- | ---------------- | ---------------- | ----------------- | --------------- | ----------- |
| PyTorch FP32 (Baseline) | 21.34             | 21.62            | 22.41            | 46.87            | 94.6689           | 46.40           | 1.0x        |
| PyTorch Dynamic INT8    | 22.15             | 22.21            | 23.48            | 45.15            | 94.6689           | 15.58           | 3.0x        |
| ONNX FP32 (Graph Fused) | 21.22             | 22.05            | 24.33            | 47.13            | 94.6689           | 15.30           | 3.0x        |
| ONNX Dynamic INT8       | 135.84            | 137.17           | 151.59           | 7.36             | 29.6718           | 4.14            | 11.2x       |

