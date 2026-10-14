"""
post_train/quantize_benchmark.py — Model Optimization & Laptop CPU Deployment Benchmarking.

Performs:
  1. PyTorch Dynamic INT8 Quantization (torch.ao.quantization)
  2. ONNX Model Export (FP32)
  3. ONNX Runtime Dynamic INT8 Quantization (onnxruntime.quantization)
  4. Real CPU Inference Benchmarking:
     - Mean latency (ms/sample)
     - Median (p50) and 95th percentile (p95) latency
     - Throughput (Frames Per Second - FPS)
     - Disk size comparison (MB)
  5. Test-set accuracy verification before vs after quantization
  6. Visual comparison plots (Latency vs Model Size)
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
from models import build_model
from config import OUTPUT_DIR, NUM_CLASSES, IMG_SIZE
from dataset import get_dataloaders
from post_train.reparameterize import reparameterize_model


def get_file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def benchmark_pytorch_latency(model: nn.Module, dummy_input: torch.Tensor, warmup: int = 20, repeats: int = 100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

    times = np.array(times)
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'fps': float(1000.0 / np.mean(times)),
    }


def benchmark_onnx_latency(onnx_path: Path, dummy_input_np: np.ndarray, warmup: int = 20, repeats: int = 100):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(onnx_path), sess_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input_np})

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input_np})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times)
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'fps': float(1000.0 / np.mean(times)),
    }


def evaluate_pytorch_accuracy(model: nn.Module, test_loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return float(correct / total) if total > 0 else 0.0


def evaluate_onnx_accuracy(onnx_path: Path, test_loader: DataLoader):
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    correct = 0
    total = 0
    for imgs, labels, _ in test_loader:
        imgs_np = imgs.numpy()
        labels_np = labels.numpy()
        outputs = session.run(None, {input_name: imgs_np})
        preds = np.argmax(outputs[0], axis=1)
        correct += int((preds == labels_np).sum())
        total += len(labels_np)
    return float(correct / total) if total > 0 else 0.0


def optimize_and_benchmark(model_name: str, run_name: str = None, num_bench_samples: int = 100):
    if run_name is None:
        tag = "adamw"
        run_name = f"{model_name}_{tag}"

    run_dir = OUTPUT_DIR / run_name
    ckpt_dir = run_dir / "checkpoints"
    plot_dir = run_dir / "plots"
    metr_dir = run_dir / "metrics"
    for d in [ckpt_dir, plot_dir, metr_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "best_model.pth"
    if not ckpt_path.exists():
        print(f"⚠️ Checkpoint not found at {ckpt_path}. Skipping optimization benchmark.")
        return None

    print("\n" + "=" * 70)
    print(f"🚀 Laptop Deployment Optimization & Benchmark: {model_name} ({run_name})")
    print("=" * 70)

    device = torch.device("cpu")
    model_fp32 = build_model(model_name, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    model_fp32.load_state_dict(ckpt['state_dict'])
    model_fp32.eval()

    # Load FULL test dataloader for accuracy evaluation.
    # Using test_run=True (200 samples) caused ONNX INT8 accuracy ≈ 15% (near-random)
    # because the tiny sample was unrepresentative. Latency benchmarking uses a
    # single dummy tensor so speed is unaffected by this change.
    _, test_loader, _, _, _ = get_dataloaders(test_run=False)

    dummy_tensor = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    dummy_np = dummy_tensor.numpy()

    results = {}

    # 1. PyTorch FP32 Baseline
    print("\n[1/5] Benchmarking PyTorch FP32 Baseline...")
    pt_fp32_lat = benchmark_pytorch_latency(model_fp32, dummy_tensor, repeats=num_bench_samples)
    pt_fp32_acc = evaluate_pytorch_accuracy(model_fp32, test_loader, device)
    pt_fp32_size = get_file_size_mb(ckpt_path)
    results['PyTorch_FP32'] = {
        **pt_fp32_lat,
        'accuracy': pt_fp32_acc,
        'size_mb': pt_fp32_size,
    }
    print(f"  Latency: {pt_fp32_lat['mean_ms']:.2f} ms | FPS: {pt_fp32_lat['fps']:.1f} | Acc: {pt_fp32_acc*100:.2f}% | Size: {pt_fp32_size:.1f} MB")

    # 2. PyTorch Reparameterized (if applicable)
    is_reparam = model_name in ['repvit_m1', 'mobileone_s2']
    if is_reparam:
        print("\n[2/5] Benchmarking PyTorch Reparameterized / Fused Model...")
        fused_path = ckpt_dir / "fused_model.pth"
        if fused_path.exists():
            fused_ckpt = torch.load(fused_path, map_location=device)
            model_fused = build_model(model_name, pretrained=False)
            model_fused = reparameterize_model(model_name, model_fused)
            model_fused.load_state_dict(fused_ckpt['state_dict'], strict=False)
        else:
            model_fused = reparameterize_model(model_name, model_fp32)
        model_fused.eval()

        pt_fused_lat = benchmark_pytorch_latency(model_fused, dummy_tensor, repeats=num_bench_samples)
        pt_fused_acc = evaluate_pytorch_accuracy(model_fused, test_loader, device)
        pt_fused_size = get_file_size_mb(fused_path) if fused_path.exists() else pt_fp32_size
        results['PyTorch_Fused'] = {
            **pt_fused_lat,
            'accuracy': pt_fused_acc,
            'size_mb': pt_fused_size,
        }
        print(f"  Latency: {pt_fused_lat['mean_ms']:.2f} ms | FPS: {pt_fused_lat['fps']:.1f} | Acc: {pt_fused_acc*100:.2f}% | Size: {pt_fused_size:.1f} MB")
        eval_base_model = model_fused
    else:
        eval_base_model = model_fp32

    # 3. PyTorch Dynamic INT8 Quantization
    print("\n[3/5] Benchmarking PyTorch Dynamic INT8 Quantization...")
    try:
        model_int8 = torch.ao.quantization.quantize_dynamic(
            eval_base_model, {nn.Linear}, dtype=torch.qint8
        )
        pt_int8_lat = benchmark_pytorch_latency(model_int8, dummy_tensor, repeats=num_bench_samples)
        pt_int8_acc = evaluate_pytorch_accuracy(model_int8, test_loader, device)
        pt_int8_ckpt = ckpt_dir / "model_int8_pt.pth"
        torch.save(model_int8.state_dict(), pt_int8_ckpt)
        pt_int8_size = get_file_size_mb(pt_int8_ckpt)
        results['PyTorch_INT8'] = {
            **pt_int8_lat,
            'accuracy': pt_int8_acc,
            'size_mb': pt_int8_size,
        }
        print(f"  Latency: {pt_int8_lat['mean_ms']:.2f} ms | FPS: {pt_int8_lat['fps']:.1f} | Acc: {pt_int8_acc*100:.2f}% | Size: {pt_int8_size:.1f} MB")
    except Exception as e:
        print(f"  PyTorch dynamic quantization failed: {e}")

    # 4. ONNX FP32 Export & Benchmark
    print("\n[4/5] Exporting & Benchmarking ONNX FP32...")
    onnx_fp32_path = ckpt_dir / f"{model_name}_fp32.onnx"
    try:
        torch.onnx.export(
            eval_base_model,
            dummy_tensor,
            str(onnx_fp32_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['logits'],
            dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
            dynamo=False,
        )
        onnx_fp32_lat = benchmark_onnx_latency(onnx_fp32_path, dummy_np, repeats=num_bench_samples)
        onnx_fp32_acc = evaluate_onnx_accuracy(onnx_fp32_path, test_loader)
        onnx_fp32_size = get_file_size_mb(onnx_fp32_path)
        results['ONNX_FP32'] = {
            **onnx_fp32_lat,
            'accuracy': onnx_fp32_acc,
            'size_mb': onnx_fp32_size,
        }
        print(f"  Latency: {onnx_fp32_lat['mean_ms']:.2f} ms | FPS: {onnx_fp32_lat['fps']:.1f} | Acc: {onnx_fp32_acc*100:.2f}% | Size: {onnx_fp32_size:.1f} MB")
    except Exception as e:
        print(f"  ONNX export failed: {e}")

    # 5. ONNX Dynamic INT8 Quantization
    print("\n[5/5] Quantizing & Benchmarking ONNX INT8...")
    onnx_int8_path = ckpt_dir / f"{model_name}_int8.onnx"
    try:
        if onnx_fp32_path.exists():
            quantize_dynamic(
                model_input=str(onnx_fp32_path),
                model_output=str(onnx_int8_path),
                weight_type=QuantType.QInt8,
            )
            onnx_int8_lat = benchmark_onnx_latency(onnx_int8_path, dummy_np, repeats=num_bench_samples)
            onnx_int8_acc = evaluate_onnx_accuracy(onnx_int8_path, test_loader)
            onnx_int8_size = get_file_size_mb(onnx_int8_path)
            results['ONNX_INT8'] = {
                **onnx_int8_lat,
                'accuracy': onnx_int8_acc,
                'size_mb': onnx_int8_size,
            }
            print(f"  Latency: {onnx_int8_lat['mean_ms']:.2f} ms | FPS: {onnx_int8_lat['fps']:.1f} | Acc: {onnx_int8_acc*100:.2f}% | Size: {onnx_int8_size:.1f} MB")
    except Exception as e:
        print(f"  ONNX dynamic quantization failed: {e}")

    # Save benchmark json
    bench_file = metr_dir / "deployment_benchmark.json"
    with open(bench_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Benchmark results saved to: {bench_file}")

    # Plot Latency vs Model Size comparison
    plot_deployment_comparison(results, model_name, plot_dir / "deployment_optimization.png")

    return results


def plot_deployment_comparison(results: dict, model_name: str, save_path: Path):
    if not results:
        return
    variants = list(results.keys())
    latencies = [results[v]['mean_ms'] for v in variants]
    sizes = [results[v]['size_mb'] for v in variants]
    fps_vals = [results[v]['fps'] for v in variants]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2b5c8f', '#3b82f6', '#10b981', '#f59e0b', '#ef4444'][:len(variants)]

    # Latency & FPS Bar chart
    bars = ax1.bar(variants, latencies, color=colors, alpha=0.85, edgecolor='black')
    ax1.set_ylabel('Inference Latency (ms/sample)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{model_name} — CPU Inference Latency (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.set_xticks(range(len(variants)))
    ax1.set_xticklabels(variants, rotation=20, ha='right')

    for bar, fps in zip(bars, fps_vals):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f} ms\n({fps:.0f} FPS)",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Size comparison Bar chart
    bars2 = ax2.bar(variants, sizes, color=colors, alpha=0.85, edgecolor='black')
    ax2.set_ylabel('Model Disk Size (MB)', fontsize=11, fontweight='bold')
    ax2.set_title(f'{model_name} — Model Footprint Reduction (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.set_xticks(range(len(variants)))
    ax2.set_xticklabels(variants, rotation=20, ha='right')

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f} MB",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved deployment comparison plot: {save_path.name}")


if __name__ == "__main__":
    from config import MODELS
    for m in [cfg['name'] for cfg in MODELS]:
        optimize_and_benchmark(m, num_bench_samples=30)
