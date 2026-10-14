"""
post_train/temperature_scaling.py — Post-Hoc Probability Calibration for Medical AI.

Implements Guo et al. (ICML 2017) Temperature Scaling:
  - Takes validation logits & labels
  - Fits scalar temperature T > 0 by minimizing NLL using L-BFGS
  - Applies T to test logits
  - Evaluates Expected Calibration Error (ECE) and Max Calibration Error (MCE)
  - Plots side-by-side reliability diagrams (Uncalibrated vs Calibrated)
  - Saves calibrated probabilities to features/test_probs_calibrated.npy
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
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
from config import OUTPUT_DIR, NUM_CLASSES, IDX_TO_CLASS


class ModelWithTemperature(nn.Module):
    """
    Decorator module that wraps logits with learned temperature T > 0.

    Reparameterisation: we optimise log_T (unconstrained) and expose
    T* = exp(log_T), which is always strictly positive.  This avoids the
    gradient instability that arises when clamping a raw parameter —
    L-BFGS would otherwise find negative raw values that are silently
    clamped to 1e-3, making the reported T* misleadingly negative.
    """
    def __init__(self):
        super().__init__()
        # log_T = 0  =>  T = exp(0) = 1.0  (identity calibration at init)
        self.log_T = nn.Parameter(torch.tensor(0.4055))  # log(1.5)

    @property
    def temperature(self) -> torch.Tensor:
        """Always-positive effective temperature."""
        return torch.exp(self.log_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """
    Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies  = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_stats = []

    for i in range(n_bins):
        b_low  = bin_boundaries[i]
        b_high = bin_boundaries[i + 1]
        in_bin = (confidences > b_low) & (confidences <= b_high)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin  = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            gap = np.abs(acc_in_bin - conf_in_bin)
            ece += gap * prop_in_bin
            mce = max(mce, gap)
            bin_stats.append({
                'bin': i,
                'range': (float(b_low), float(b_high)),
                'acc': float(acc_in_bin),
                'conf': float(conf_in_bin),
                'count': int(np.sum(in_bin)),
            })
        else:
            bin_stats.append({
                'bin': i,
                'range': (float(b_low), float(b_high)),
                'acc': 0.0,
                'conf': 0.0,
                'count': 0,
            })

    return float(ece), float(mce), bin_stats


def calibrate_model(model_name: str, run_name: str = None):
    if run_name is None:
        tag = "adamw"
        run_name = f"{model_name}_{tag}"

    run_dir = OUTPUT_DIR / run_name
    feat_dir = run_dir / "features"
    plot_dir = run_dir / "plots"
    metr_dir = run_dir / "metrics"
    for d in [feat_dir, plot_dir, metr_dir]:
        d.mkdir(parents=True, exist_ok=True)

    test_logits_path = feat_dir / "test_logits.npy"
    test_labels_path = feat_dir / "test_labels.npy"

    if not test_logits_path.exists() or not test_labels_path.exists():
        print(f"⚠️ Feature logits not found in {feat_dir}. Run feature extraction first.")
        return None

    print("\n" + "=" * 70)
    print(f"🌡️ Temperature Scaling Calibration: {model_name} ({run_name})")
    print("=" * 70)

    test_logits_np = np.load(test_logits_path)
    test_labels_np = np.load(test_labels_path)

    # ── Calibration protocol (Guo et al., ICML 2017) ────────────────────────
    # We must NOT use train logits — the model memorised training data, so
    # train logits are over-confident (high max-softmax) and L-BFGS would
    # drive T → large negative values to compensate, which is physically
    # meaningless (T must be > 0).
    #
    # Correct approach: hold out the FIRST half of the test set as the
    # calibration set, and evaluate ECE / NLL on the SECOND half.
    # This matches the standard train/val/test split spirit even when only
    # train and test CSVs exist.
    rng = np.random.default_rng(seed=42)
    n_test = len(test_logits_np)
    idx_all = rng.permutation(n_test)
    calib_idx = idx_all[: n_test // 2]
    eval_idx  = idx_all[n_test // 2 :]

    calib_logits_np = test_logits_np[calib_idx]
    calib_labels_np = test_labels_np[calib_idx]
    eval_logits_np  = test_logits_np[eval_idx]
    eval_labels_np  = test_labels_np[eval_idx]

    print(f"  Calibration set : {len(calib_idx)} samples")
    print(f"  Evaluation set  : {len(eval_idx)} samples")

    calib_logits = torch.from_numpy(calib_logits_np).float()
    calib_labels = torch.from_numpy(calib_labels_np).long()
    test_logits  = torch.from_numpy(eval_logits_np).float()
    test_labels  = torch.from_numpy(eval_labels_np).long()
    test_labels_np = eval_labels_np   # rebind for ECE computation below

    # Uncalibrated baseline
    probs_uncal = F.softmax(test_logits, dim=1).numpy()
    ece_before, mce_before, _ = compute_ece(probs_uncal, test_labels_np)
    nll_criterion = nn.CrossEntropyLoss()
    nll_before = nll_criterion(test_logits, test_labels).item()

    print(f"  Uncalibrated NLL : {nll_before:.4f}")
    print(f"  Uncalibrated ECE : {ece_before:.4f} ({ece_before*100:.2f}%)")
    print(f"  Uncalibrated MCE : {mce_before:.4f}")

    # Optimize log_T using L-BFGS (log_T is unconstrained; T = exp(log_T) > 0)
    temp_model = ModelWithTemperature()
    optimizer = torch.optim.LBFGS([temp_model.log_T], lr=0.01, max_iter=100)

    def eval_loss():
        optimizer.zero_grad()
        loss = nll_criterion(temp_model(calib_logits), calib_labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    optimal_T = float(temp_model.temperature.item())   # always positive
    print(f"  Optimal Temperature T* : {optimal_T:.4f}")

    # Calibrated probabilities
    with torch.no_grad():
        calibrated_test_logits = temp_model(test_logits)
        probs_cal = F.softmax(calibrated_test_logits, dim=1).numpy()
        nll_after = nll_criterion(calibrated_test_logits, test_labels).item()

    ece_after, mce_after, _ = compute_ece(probs_cal, test_labels_np)
    print(f"  Calibrated NLL   : {nll_after:.4f}")
    print(f"  Calibrated ECE   : {ece_after:.4f} ({ece_after*100:.2f}%)")
    print(f"  Calibrated MCE   : {mce_after:.4f}")
    ece_reduction = (ece_before - ece_after) / max(ece_before, 1e-6) * 100
    print(f"  ECE Relative Drop: {ece_reduction:.1f}%")

    # Save calibrated probabilities
    cal_probs_path = feat_dir / "test_probs_calibrated.npy"
    np.save(cal_probs_path, probs_cal)
    print(f"  💾 Saved calibrated probabilities: {cal_probs_path.name}")

    # Save calibration stats
    calib_stats = {
        'model_name': model_name,
        'run_name': run_name,
        'optimal_temperature': optimal_T,
        'nll_before': float(nll_before),
        'nll_after': float(nll_after),
        'ece_before': float(ece_before),
        'ece_after': float(ece_after),
        'mce_before': float(mce_before),
        'mce_after': float(mce_after),
        'ece_relative_reduction_pct': float(ece_reduction),
    }
    stats_path = metr_dir / "calibration_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(calib_stats, f, indent=2)
    print(f"  📊 Saved calibration stats: {stats_path.name}")

    # Plot comparison reliability diagram
    plot_calibration_comparison(
        probs_uncal, probs_cal, test_labels_np,
        optimal_T, ece_before, ece_after,
        model_name, plot_dir / "calibration_temperature_scaling.png"
    )

    return calib_stats


def plot_calibration_comparison(probs_uncal, probs_cal, labels, T, ece_before, ece_after, model_name, save_path):
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, probs, title, ece_val, color in [
        (ax1, probs_uncal, f'Uncalibrated (T=1.00)\nECE = {ece_before*100:.2f}%', ece_before, '#ef4444'),
        (ax2, probs_cal,   f'Temperature Scaled (T*={T:.2f})\nECE = {ece_after*100:.2f}%', ece_after, '#10b981'),
    ]:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies  = (predictions == labels).astype(float)

        bin_accs  = []
        bin_confs = []
        bin_counts = []
        for i in range(n_bins):
            in_b = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
            if np.sum(in_b) > 0:
                bin_accs.append(np.mean(accuracies[in_b]))
                bin_confs.append(np.mean(confidences[in_b]))
                bin_counts.append(np.sum(in_b))
            else:
                bin_accs.append(0.0)
                bin_confs.append((bin_edges[i] + bin_edges[i+1])/2)
                bin_counts.append(0)

        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5)
        ax.bar(centers, bin_accs, width=1.0/n_bins, color=color, alpha=0.65, edgecolor='black', label='Outputs')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(f"{model_name}: {title}", fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved calibration comparison plot: {save_path.name}")


if __name__ == "__main__":
    from config import MODELS
    for m in [cfg['name'] for cfg in MODELS]:
        calibrate_model(m)
