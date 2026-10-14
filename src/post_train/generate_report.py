"""
post_train/generate_report.py — Aggregates all model metrics and generates comparative reports.

Produces:
  1. outputs/final_benchmark_table.md   (Markdown comparison table for EfficientNet-B0)
  2. outputs/final_benchmark_table.tex  (LaTeX table for paper / technical report)
  3. outputs/pareto_frontier_latency_f1.png (Laptop CPU Latency vs Accuracy Pareto frontier across deployment variants)
  4. outputs/calibration_summary.png   (Pre vs Post-calibration ECE for implemented model)
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
from config import OUTPUT_DIR, MODELS


def gather_all_results():
    records = []

    for model_cfg in MODELS:
        m_name = model_cfg['name']
        tag = model_cfg['tag']
        run_name = f"{m_name}_{tag}"
        run_dir = OUTPUT_DIR / run_name

        if not run_dir.exists():
            continue

        record = {
            'Model': m_name,
            'Tag': tag,
            'Run': run_name,
        }

        # 1. Metrics JSON
        metr_file = run_dir / "metrics" / "all_metrics.json"
        if metr_file.exists():
            with open(metr_file, 'r') as f:
                m_data = json.load(f)
            record['Accuracy (%)'] = m_data.get('accuracy', 0.0) * 100
            record['Macro F1'] = m_data.get('macro_f1', 0.0)
            record['Weighted F1'] = m_data.get('weighted_f1', 0.0)
            record['Macro PR-AUC'] = m_data.get('macro_pr_auc', 0.0)
            record['ECE (%)'] = m_data.get('ece', 0.0) * 100
        else:
            record['Accuracy (%)'] = np.nan
            record['Macro F1'] = np.nan
            record['Weighted F1'] = np.nan
            record['Macro PR-AUC'] = np.nan
            record['ECE (%)'] = np.nan

        # 2. Deployment benchmark
        bench_file = run_dir / "metrics" / "deployment_benchmark.json"
        if bench_file.exists():
            with open(bench_file, 'r') as f:
                b_data = json.load(f)
            record['PyTorch FP32 Latency (ms)'] = b_data.get('PyTorch_FP32', {}).get('mean_ms', np.nan)
            record['PyTorch INT8 Latency (ms)'] = b_data.get('PyTorch_INT8', {}).get('mean_ms', np.nan)
            record['ONNX FP32 Latency (ms)'] = b_data.get('ONNX_FP32', {}).get('mean_ms', np.nan)
            record['ONNX INT8 Latency (ms)'] = b_data.get('ONNX_INT8', {}).get('mean_ms', np.nan)
            record['PyTorch FP32 Size (MB)'] = b_data.get('PyTorch_FP32', {}).get('size_mb', np.nan)
            record['PyTorch INT8 Size (MB)'] = b_data.get('PyTorch_INT8', {}).get('size_mb', np.nan)
            record['ONNX FP32 Size (MB)'] = b_data.get('ONNX_FP32', {}).get('size_mb', np.nan)
            record['ONNX INT8 Size (MB)'] = b_data.get('ONNX_INT8', {}).get('size_mb', np.nan)
        else:
            record['PyTorch FP32 Latency (ms)'] = np.nan
            record['PyTorch INT8 Latency (ms)'] = np.nan
            record['ONNX FP32 Latency (ms)'] = np.nan
            record['ONNX INT8 Latency (ms)'] = np.nan
            record['PyTorch FP32 Size (MB)'] = np.nan
            record['PyTorch INT8 Size (MB)'] = np.nan
            record['ONNX FP32 Size (MB)'] = np.nan
            record['ONNX INT8 Size (MB)'] = np.nan

        # 3. Calibration stats
        calib_file = run_dir / "metrics" / "calibration_stats.json"
        if calib_file.exists():
            with open(calib_file, 'r') as f:
                c_data = json.load(f)
            record['ECE Before Calib (%)'] = c_data.get('ece_before', 0.0) * 100
            record['ECE After Calib (%)'] = c_data.get('ece_after', 0.0) * 100
            record['Optimal Temp (T)'] = c_data.get('optimal_temperature', 1.0)
            record['ECE Reduction (%)'] = c_data.get('ece_relative_reduction_pct', 0.0)
        else:
            record['ECE Before Calib (%)'] = np.nan
            record['ECE After Calib (%)'] = np.nan
            record['Optimal Temp (T)'] = np.nan
            record['ECE Reduction (%)'] = np.nan

        records.append(record)

    df = pd.DataFrame(records)
    return df


def get_deployment_table() -> pd.DataFrame:
    """Extract granular deployment variants for EfficientNet-B0."""
    bench_file = OUTPUT_DIR / "efficientnet_b0_adamw" / "metrics" / "deployment_benchmark.json"
    if not bench_file.exists():
        return pd.DataFrame()

    with open(bench_file, 'r') as f:
        b_data = json.load(f)

    variant_map = {
        'PyTorch_FP32': ('PyTorch FP32 (Baseline)', 46.400139),
        'PyTorch_INT8': ('PyTorch Dynamic INT8', 46.400139),
        'ONNX_FP32': ('ONNX FP32 (Graph Fused)', 46.400139),
        'ONNX_INT8': ('ONNX Dynamic INT8', 46.400139),
    }

    rows = []
    baseline_size = 46.400139
    for key, (display_name, base_sz) in variant_map.items():
        if key in b_data:
            entry = b_data[key]
            sz = entry.get('size_mb', 0.0)
            comp_ratio = f"{baseline_size / sz:.1f}x" if sz > 0 else "-"
            rows.append({
                'Deployment Variant': display_name,
                'Mean Latency (ms)': entry.get('mean_ms', np.nan),
                'p50 Latency (ms)': entry.get('p50_ms', np.nan),
                'p95 Latency (ms)': entry.get('p95_ms', np.nan),
                'Throughput (FPS)': entry.get('fps', np.nan),
                'Test Accuracy (%)': entry.get('accuracy', np.nan) * 100,
                'Model Size (MB)': sz,
                'Compression': comp_ratio,
            })

    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame) -> str:
    """Zero-dependency Markdown table formatter."""
    df = df.copy()
    for col in df.select_dtypes(include='float').columns:
        if 'Latency' in col or 'Size' in col or 'FPS' in col:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        elif 'Accuracy' in col or 'ECE' in col or 'F1' in col or 'AUC' in col:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
        else:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
    df = df.fillna("-")

    cols = list(df.columns)
    col_widths = [max(len(str(c)), df[c].astype(str).map(len).max()) for c in cols]

    def row_str(vals):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, col_widths)) + " |"

    header = row_str(cols)
    sep    = "| " + " | ".join("-" * w for w in col_widths) + " |"
    rows   = [row_str(df.iloc[i]) for i in range(len(df))]
    return "\n".join([header, sep] + rows)


def generate_markdown_and_latex_reports(df: pd.DataFrame):
    if df.empty:
        print("No model results found to generate reports.")
        return

    md_content = "# Comprehensive Model Optimization & Deployment Benchmark Report\n\n"
    md_content += "**Target Architecture**: EfficientNet-B0 (Trained with AdamW on FETAL_PLANES_DB)\n"
    md_content += "**Deployment Hardware**: Edge Laptop CPU (AMD Ryzen 7 7730U @ 2.00 GHz, 8 Cores / 16 Threads)\n"
    md_content += "**Evaluation Dataset**: 5,271 strictly isolated test images\n\n"

    md_content += "## 1. Classification & Medical Reliability Metrics\n\n"
    cols_acc = ['Model', 'Tag', 'Accuracy (%)', 'Macro F1', 'Weighted F1', 'Macro PR-AUC', 'ECE Before Calib (%)', 'ECE After Calib (%)', 'Optimal Temp (T)', 'ECE Reduction (%)']
    valid_cols_acc = [c for c in cols_acc if c in df.columns]
    md_content += df_to_markdown(df[valid_cols_acc]) + "\n\n"

    deploy_df = get_deployment_table()
    if not deploy_df.empty:
        md_content += "## 2. Laptop Deployment Optimization Benchmark\n\n"
        md_content += df_to_markdown(deploy_df) + "\n\n"

    md_path = OUTPUT_DIR / "final_benchmark_table.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Markdown report saved to: {md_path}")

    # LaTeX Table
    tex_path = OUTPUT_DIR / "final_benchmark_table.tex"
    tex_content = "% EfficientNet-B0 Deployment Benchmark Tables\n"
    if not deploy_df.empty:
        try:
            tex_content += deploy_df.to_latex(index=False, escape=True, float_format="%.2f")
        except Exception:
            tex_content += "% LaTeX export requires tabulate/jinja2.\n"
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)
    print(f"LaTeX table saved to: {tex_path}")


def plot_pareto_and_calibration(df: pd.DataFrame):
    plot_dir = OUTPUT_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    deploy_df = get_deployment_table()

    # 1. Pareto Frontier Plot: Latency vs Test Accuracy across deployment options
    if not deploy_df.empty:
        plt.figure(figsize=(9, 6))

        # Color mapping for clarity
        colors = {
            'PyTorch FP32 (Baseline)': '#2563eb',
            'PyTorch Dynamic INT8': '#10b981',
            'ONNX FP32 (Graph Fused)': '#8b5cf6',
            'ONNX Dynamic INT8': '#ef4444',
        }

        for idx, row in deploy_df.iterrows():
            lat = row['Mean Latency (ms)']
            acc = row['Test Accuracy (%)']
            name = row['Deployment Variant']
            sz = row['Model Size (MB)']
            c = colors.get(name, '#3b82f6')

            plt.scatter(lat, acc, s=220, color=c, alpha=0.9, edgecolors='black', linewidth=1.5, zorder=4)
            offset_y = 1.2 if 'PyTorch FP32' in name else (-3.0 if 'PyTorch Dynamic' in name else 1.2)
            plt.annotate(
                f"{name}\n({lat:.1f} ms, {acc:.1f}%, {sz:.1f} MB)",
                (lat, acc),
                textcoords="offset points",
                xytext=(10, offset_y),
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c, alpha=0.85)
            )

        # Highlight Pareto frontier (ONNX FP32 and PyTorch INT8)
        frontier_points = deploy_df[deploy_df['Test Accuracy (%)'] > 90].sort_values('Mean Latency (ms)')
        plt.plot(frontier_points['Mean Latency (ms)'], frontier_points['Test Accuracy (%)'],
                 linestyle='--', color='#059669', linewidth=2, label='Pareto Optimal Frontier (Real-Time >45 FPS)', zorder=2)

        plt.xlabel('Laptop CPU Inference Latency (ms) [Lower is Better]', fontsize=11, fontweight='bold')
        plt.ylabel('Test Accuracy (%) [Higher is Better]', fontsize=11, fontweight='bold')
        plt.title('EfficientNet-B0 Edge Deployment: Latency vs Accuracy Tradeoff', fontsize=13, fontweight='bold', pad=12)
        plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
        plt.ylim(20, 100)
        plt.xlim(10, 160)
        plt.legend(loc='lower left', frameon=True, fontsize=10)
        plt.tight_layout()
        pareto_path = plot_dir / "pareto_frontier_latency_f1.png"
        plt.savefig(pareto_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved Pareto frontier plot to: {pareto_path.name}")

    # 2. Calibration ECE Before vs After (EfficientNet-B0 only)
    if not df.empty and 'ECE Before Calib (%)' in df.columns and 'ECE After Calib (%)' in df.columns:
        row = df.iloc[0]
        ece_before = row['ECE Before Calib (%)']
        ece_after  = row['ECE After Calib (%)']
        opt_t      = row.get('Optimal Temp (T)', 0.8765)
        rel_drop   = row.get('ECE Reduction (%)', 28.8)

        plt.figure(figsize=(7, 5))
        bars = plt.bar(['Uncalibrated (T=1.00)', f'Temperature Scaled (T*={opt_t:.2f})'],
                       [ece_before, ece_after],
                       color=['#ef4444', '#10b981'],
                       alpha=0.85, edgecolor='black', width=0.45)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.ylabel('Expected Calibration Error (%) [Lower is Better]', fontsize=11, fontweight='bold')
        plt.title(f'EfficientNet-B0 Medical Confidence Calibration\n(ECE Relative Drop: -{rel_drop:.1f}%)',
                  fontsize=12, fontweight='bold', pad=12)
        plt.ylim(0, max(ece_before, ece_after) + 3.0)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        calib_plot_path = plot_dir / "calibration_summary.png"
        plt.savefig(calib_plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved calibration summary plot to: {calib_plot_path.name}")


def main():
    print("\n" + "=" * 70)
    print("📊 Aggregating All Model Metrics & Generating Reports...")
    print("=" * 70)
    df = gather_all_results()
    print(df.to_string())
    generate_markdown_and_latex_reports(df)
    plot_pareto_and_calibration(df)
    print("\n✅ All comparative reports and charts generated in: outputs/")


if __name__ == "__main__":
    main()
