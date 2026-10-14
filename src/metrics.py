"""
Metrics computation for FETAL_PLANES_DB.
Computes and returns:
  - Accuracy (overall + per-class)
  - Macro F1
  - Weighted F1
  - Per-class PR-AUC (+ macro average)
  - ECE (Expected Calibration Error)  — critical for medical AI
  - Per-operator accuracy
  - Per-machine accuracy
  - Confusion matrix (raw counts)
All functions take numpy arrays for portability (no torch dependency at eval time).
"""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    average_precision_score, classification_report
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd

from config import IDX_TO_CLASS, CLASSES_6, NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error.
    probs  : [N, C] softmax probabilities
    labels : [N]    true class indices
    """
    confidences = probs.max(axis=1)          # [N] max softmax score
    predictions = probs.argmax(axis=1)       # [N] predicted class
    correct     = (predictions == labels).astype(float)

    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc_bin  = correct[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += mask.sum() * abs(acc_bin - conf_bin)

    return float(ece / len(labels))


def compute_all_metrics(
    probs       : np.ndarray,    # [N, C] softmax probs
    preds       : np.ndarray,    # [N]    predicted classes
    labels      : np.ndarray,    # [N]    true classes
    meta_df     : pd.DataFrame,  # DataFrame with columns: operator, machine
) -> dict:
    """
    Full metric suite. Returns a nested dict.
    meta_df index must align with probs/preds/labels.
    """
    metrics = {}

    # ── Overall ──────────────────────────────────────────────────────────────
    metrics['accuracy']    = float(accuracy_score(labels, preds))
    metrics['macro_f1']    = float(f1_score(labels, preds, average='macro',   zero_division=0))
    metrics['weighted_f1'] = float(f1_score(labels, preds, average='weighted', zero_division=0))
    metrics['ece']         = compute_ece(probs, labels)

    # ── PR-AUC per class (one-vs-rest) ───────────────────────────────────────
    pr_auc_per_class = {}
    labels_onehot = np.eye(NUM_CLASSES)[labels]
    for c in range(NUM_CLASSES):
        try:
            auc_c = float(average_precision_score(labels_onehot[:, c], probs[:, c]))
        except Exception:
            auc_c = 0.0
        pr_auc_per_class[IDX_TO_CLASS[c]] = auc_c
    metrics['pr_auc_per_class'] = pr_auc_per_class
    metrics['macro_pr_auc']     = float(np.mean(list(pr_auc_per_class.values())))

    # ── Per-class accuracy ───────────────────────────────────────────────────
    per_class_acc = {}
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() == 0:
            per_class_acc[IDX_TO_CLASS[c]] = None
            continue
        per_class_acc[IDX_TO_CLASS[c]] = float(accuracy_score(labels[mask], preds[mask]))
    metrics['per_class_accuracy'] = per_class_acc

    # ── Per-operator accuracy ─────────────────────────────────────────────────
    if meta_df is not None and 'operator' in meta_df.columns:
        meta_df = meta_df.reset_index(drop=True)
        per_op = {}
        for op, grp in meta_df.groupby('operator'):
            idx = grp.index.values
            per_op[op] = float(accuracy_score(labels[idx], preds[idx]))
        metrics['per_operator_accuracy'] = per_op

    # ── Per-machine accuracy ──────────────────────────────────────────────────
    if meta_df is not None and 'machine' in meta_df.columns:
        per_mach = {}
        for mach, grp in meta_df.groupby('machine'):
            idx = grp.index.values
            per_mach[mach] = float(accuracy_score(labels[idx], preds[idx]))
        metrics['per_machine_accuracy'] = per_mach

    # ── Classification report (printed, not stored in dict) ──────────────────
    class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)
    metrics['classification_report'] = report

    return metrics


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, save_path: Path, title: str = "Confusion Matrix"):
    class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{title} — Raw Counts")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    axes[0].tick_params(axis='x', rotation=45)

    # Normalised
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"{title} — Row-Normalised")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Confusion matrix saved → {save_path}")


def plot_training_curves(history: dict, save_path: Path, title: str = "Training Curves"):
    """
    history: dict with keys like 'train_loss', 'val_loss',
             'train_acc', 'val_acc', 'train_f1', 'val_f1', etc.
    """
    epochs = list(range(1, len(history['train_loss']) + 1))

    metrics_to_plot = [
        ('train_loss',    'val_loss',    'Loss',     'Loss'),
        ('train_acc',     'val_acc',     'Accuracy', 'Accuracy (%)'),
        ('train_macro_f1','val_macro_f1','Macro F1', 'F1 Score'),
    ]

    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))

    for ax, (train_key, val_key, label, ylabel) in zip(axes, metrics_to_plot):
        if train_key in history:
            ax.plot(epochs, history[train_key], label=f'Train', color='steelblue', linewidth=2)
        if val_key in history:
            ax.plot(epochs, history[val_key],   label=f'Val',   color='darkorange',
                    linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Training curves saved → {save_path}")


def plot_per_class_metrics(metrics: dict, save_path: Path, title: str = "Per-Class Metrics"):
    class_names = list(metrics['pr_auc_per_class'].keys())

    pr_aucs  = [metrics['pr_auc_per_class'][c]    for c in class_names]
    accs     = [metrics['per_class_accuracy'][c]  for c in class_names]

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, accs,    width, label='Accuracy',  color='steelblue',  alpha=0.85)
    bars2 = ax.bar(x + width/2, pr_aucs, width, label='PR-AUC',    color='darkorange', alpha=0.85)

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(f"{title}\n"
                 f"Overall Acc={metrics['accuracy']:.3f} | "
                 f"Macro-F1={metrics['macro_f1']:.3f} | "
                 f"ECE={metrics['ece']:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Value labels on bars
    for bar in bars1 + bars2:
        h = bar.get_height()
        if h is not None:
            ax.annotate(f'{h:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Per-class metrics saved → {save_path}")


def plot_reliability_diagram(probs, labels, save_path: Path, title: str = "Reliability Diagram"):
    """Calibration reliability diagram (confidence vs accuracy)."""
    fig, ax = plt.subplots(figsize=(7, 7))

    n_bins = 15
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        correct, confidences, n_bins=n_bins, strategy='uniform'
    )

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    ax.plot(mean_predicted_value, fraction_of_positives,
            's-', color='steelblue', label='Model', linewidth=2, markersize=7)
    ax.fill_between(mean_predicted_value, mean_predicted_value,
                    fraction_of_positives, alpha=0.2, color='steelblue')

    ece = compute_ece(probs, labels, n_bins=n_bins)
    ax.set_xlabel('Mean Predicted Confidence')
    ax.set_ylabel('Fraction Correct')
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Reliability diagram saved → {save_path}")


def plot_per_group_accuracy(metrics: dict, save_path: Path, title: str = "Per-Group Accuracy"):
    """Bar plot of per-operator and per-machine accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, key, label in [
        (axes[0], 'per_operator_accuracy', 'Operator'),
        (axes[1], 'per_machine_accuracy',  'US Machine'),
    ]:
        if key not in metrics:
            ax.set_visible(False)
            continue
        groups = list(metrics[key].keys())
        vals   = list(metrics[key].values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        bars = ax.bar(groups, vals, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel('Accuracy')
        ax.set_title(f"Per-{label} Accuracy")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', rotation=30)
        ax.axhline(metrics['accuracy'], color='red', linestyle='--', linewidth=1.5,
                   label=f"Overall={metrics['accuracy']:.3f}")
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Per-group accuracy saved → {save_path}")


def save_metrics_json(metrics: dict, save_path: Path):
    """Save metrics dict to JSON (removes non-serialisable objects)."""
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            clean[k] = {str(kk): (float(vv) if vv is not None else None)
                        for kk, vv in v.items()}
        elif isinstance(v, (float, int)):
            clean[k] = v
        elif isinstance(v, str):
            clean[k] = v
        # skip non-serialisable
    with open(save_path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"[Metrics] Saved → {save_path}")
