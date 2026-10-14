"""
Main training script for FETAL_PLANES_DB.

Usage:
    python src/train.py --model efficientnet_b0                          # baseline
    python src/train.py --model repvit_m1   --use_asam --use_cutmix     # ASAM + CutMix
    python src/train.py --model mobileone_s2 --use_asam --use_cutmix    # ASAM + CutMix
    python src/train.py --model repvit_m1   --test_run                  # 2-epoch smoke test

Everything is saved under: outputs/{model_name}_{tag}/
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import json
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data.mixup import Mixup

# ── Local imports (must run from r:\HTIC\) ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, EPOCHS, LR, WEIGHT_DECAY, LABEL_SMOOTHING,
    PATIENCE, ASAM_RHO, ASAM_ETA, MIXUP_ALPHA, CUTMIX_ALPHA, MIXUP_PROB,
    SAVE_EVERY_N, USE_WANDB, WANDB_PROJECT, WANDB_ENTITY, SEED,
    TEST_RUN_EPOCHS, IDX_TO_CLASS, NUM_CLASSES, BATCH_SIZE,
)
from dataset          import get_dataloaders
from models           import build_model
from asam             import ASAM
from metrics          import (
    compute_all_metrics, plot_confusion_matrix, plot_training_curves,
    plot_per_class_metrics, plot_reliability_diagram,
    plot_per_group_accuracy, save_metrics_json
)
from feature_extractor import extract_and_save_features


# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train fetal plane classifier")
    parser.add_argument('--model',      type=str,  default='repvit_m1',
                        choices=['efficientnet_b0', 'repvit_m1', 'mobileone_s2'])
    parser.add_argument('--epochs',     type=int,  default=EPOCHS)
    parser.add_argument('--lr',         type=float, default=LR)
    parser.add_argument('--batch_size', type=int,  default=BATCH_SIZE)
    parser.add_argument('--use_asam',   action='store_true',
                        help='Use ASAM optimizer (default: AdamW)')
    parser.add_argument('--use_cutmix', action='store_true',
                        help='Apply CutMix + Mixup augmentation')
    parser.add_argument('--test_run',   action='store_true',
                        help='Smoke test: 2 epochs + subset of data')
    parser.add_argument('--no_wandb',   action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--seed',       type=int,  default=SEED)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model, loader, criterion, optimizer,
    device, epoch, mixup_fn,
    use_asam: bool, use_cutmix: bool,
):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    pbar = tqdm(loader, desc=f"Train E{epoch}", ncols=100, leave=False)

    for imgs, labels, metas in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply CutMix / Mixup (modifies images and produces soft labels)
        if use_cutmix and mixup_fn is not None:
            imgs, soft_labels = mixup_fn(imgs, labels)
            use_soft = True
        else:
            use_soft = False

        if use_asam:
            # ── ASAM two-step ────────────────────────────────────────────────
            optimizer.zero_grad()
            # Step 1: compute gradient at θ, move to perturbed θ + ε̂
            logits = model(imgs)
            if use_soft:
                loss = criterion(logits, soft_labels)
            else:
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.ascent_step()

            # Step 2: compute gradient at perturbed point, update θ
            logits2 = model(imgs)
            if use_soft:
                loss2 = criterion(logits2, soft_labels)
            else:
                loss2 = criterion(logits2, labels)
            loss2.backward()
            optimizer.descent_step()

            loss = loss2   # report second loss (post-perturbation)

        else:
            # ── Standard AdamW ───────────────────────────────────────────────
            optimizer.zero_grad()
            logits = model(imgs)
            if use_soft:
                loss = criterion(logits, soft_labels)
            else:
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Accuracy on hard labels (even if training used soft)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{correct/total:.3f}")

    avg_loss = total_loss / total
    avg_acc  = correct / total
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, compute_macro_f1=True):
    """
    Returns: (avg_loss, accuracy, macro_f1, all_probs_np, all_preds_np, all_labels_np, meta_df)
    """
    from sklearn.metrics import f1_score

    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_preds  = []
    all_labels = []
    all_metas  = []

    for imgs, labels, metas in tqdm(loader, desc="Eval", ncols=100, leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_metas.extend(metas)

    probs_np  = np.concatenate(all_probs,  axis=0)
    preds_np  = np.concatenate(all_preds,  axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    meta_df   = pd.DataFrame(all_metas)

    avg_loss  = total_loss / len(labels_np)
    accuracy  = float((preds_np == labels_np).mean())
    macro_f1  = float(f1_score(labels_np, preds_np, average='macro', zero_division=0))

    return avg_loss, accuracy, macro_f1, probs_np, preds_np, labels_np, meta_df


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Build run tag ────────────────────────────────────────────────────────
    tag_parts = []
    if args.use_asam:   tag_parts.append('asam')
    if args.use_cutmix: tag_parts.append('cutmix')
    if not tag_parts:   tag_parts.append('adamw')
    tag = '_'.join(tag_parts)

    run_name  = f"{args.model}_{tag}"
    run_dir   = OUTPUT_DIR / run_name
    ckpt_dir  = run_dir / "checkpoints"
    feat_dir  = run_dir / "features"
    plot_dir  = run_dir / "plots"
    metr_dir  = run_dir / "metrics"
    for d in [ckpt_dir, feat_dir, plot_dir, metr_dir]:
        d.mkdir(parents=True, exist_ok=True)

    n_epochs  = TEST_RUN_EPOCHS if args.test_run else args.epochs
    device    = torch.device("cpu")

    print("=" * 70)
    print(f" Model   : {args.model}")
    print(f" Tag     : {tag}")
    print(f" Epochs  : {n_epochs}")
    print(f" ASAM    : {args.use_asam}")
    print(f" CutMix  : {args.use_cutmix}")
    print(f" TestRun : {args.test_run}")
    print(f" Device  : {device}")
    print(f" Output  : {run_dir}")
    print("=" * 70)

    # ── W&B ─────────────────────────────────────────────────────────────────
    use_wandb = USE_WANDB and not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project = WANDB_PROJECT,
                entity  = WANDB_ENTITY,
                name    = run_name,
                config  = vars(args),
                dir     = str(run_dir),
            )
            print(f"[W&B] Run: {wandb.run.url}")
        except Exception as e:
            print(f"[W&B] Failed to init: {e} — continuing without W&B")
            use_wandb = False

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, test_loader, train_df, test_df, class_weights = get_dataloaders(
        test_run=args.test_run
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(args.model, pretrained=True)
    model = model.to(device)

    # ── Loss ─────────────────────────────────────────────────────────────────
    # Class-weighted CE with label smoothing — handles both hard and soft labels
    # For soft labels (from CutMix), we use KLDiv manually in training loop
    # For eval (hard labels), CrossEntropyLoss
    criterion_train = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=LABEL_SMOOTHING
    )
    criterion_eval = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    # ── CutMix / Mixup ───────────────────────────────────────────────────────
    mixup_fn = None
    if args.use_cutmix:
        mixup_fn = Mixup(
            mixup_alpha   = MIXUP_ALPHA,
            cutmix_alpha  = CUTMIX_ALPHA,
            prob          = MIXUP_PROB,
            switch_prob   = 0.5,       # 50% cutmix, 50% mixup
            mode          = 'batch',
            label_smoothing = LABEL_SMOOTHING,
            num_classes   = NUM_CLASSES,
        )
        print("[CutMix] Enabled — Mixup α=0.8, CutMix α=1.0, prob=0.5")

    # ── Optimizer ────────────────────────────────────────────────────────────
    base_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    if args.use_asam:
        optimizer = ASAM(base_optimizer, model, rho=ASAM_RHO, eta=ASAM_ETA)
        print(f"[Optimizer] ASAM (ρ={ASAM_RHO}, η={ASAM_ETA}) wrapping AdamW")
    else:
        optimizer = base_optimizer
        print(f"[Optimizer] AdamW (lr={args.lr}, wd={WEIGHT_DECAY})")

    # ── Scheduler ────────────────────────────────────────────────────────────
    scheduler = CosineAnnealingLR(base_optimizer, T_max=n_epochs, eta_min=1e-6)

    # ── History & early stopping ─────────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [],
        'train_macro_f1': [], 'val_macro_f1': [],
        'lr': [],
    }
    best_val_acc   = 0.0
    best_epoch     = 0
    patience_count = 0

    # ── Training loop ────────────────────────────────────────────────────────
    print("\n🚀 Starting training...\n")
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion_train, optimizer,
            device, epoch, mixup_fn,
            use_asam=args.use_asam, use_cutmix=args.use_cutmix,
        )

        # Eval
        val_loss, val_acc, val_f1, val_probs, val_preds, val_labels, val_meta = evaluate(
            model, test_loader, criterion_eval, device
        )

        # Train metrics tracked directly during training epoch
        train_f1 = train_acc

        scheduler.step()
        current_lr = base_optimizer.param_groups[0]['lr']
        elapsed    = time.time() - epoch_start

        # ── Log ──────────────────────────────────────────────────────────────
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc * 100)
        history['val_acc'].append(val_acc * 100)
        history['train_macro_f1'].append(train_f1)
        history['val_macro_f1'].append(val_f1)
        history['lr'].append(current_lr)

        print(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}%  F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%  F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.0f}s"
        )

        if use_wandb:
            try:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss, 'train/acc': train_acc * 100,
                    'train/macro_f1': train_f1,
                    'val/loss': val_loss,   'val/acc': val_acc * 100,
                    'val/macro_f1': val_f1,
                    'lr': current_lr,
                })
            except Exception:
                pass

        # ── Checkpoint: every N epochs ────────────────────────────────────────
        if epoch % SAVE_EVERY_N == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pth"
            torch.save({
                'epoch'     : epoch,
                'model_name': args.model,
                'tag'       : tag,
                'state_dict': model.state_dict(),
                'val_acc'   : val_acc,
                'val_f1'    : val_f1,
            }, ckpt_path)
            print(f"  [Ckpt] Periodic checkpoint → {ckpt_path.name}")

        # ── Checkpoint: best model ────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_epoch     = epoch
            patience_count = 0
            ckpt_path = ckpt_dir / "best_model.pth"
            torch.save({
                'epoch'     : epoch,
                'model_name': args.model,
                'tag'       : tag,
                'state_dict': model.state_dict(),
                'optimizer_state': base_optimizer.state_dict(),
                'val_acc'   : val_acc,
                'val_f1'    : val_f1,
                'val_loss'  : val_loss,
            }, ckpt_path)
            print(f"  ✅ New best val acc: {val_acc*100:.2f}%  → saved best_model.pth")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n[EarlyStopping] No improvement for {PATIENCE} epochs. Stopping.")
                break

    # ── Save last epoch ───────────────────────────────────────────────────────
    torch.save({
        'epoch'     : epoch,
        'model_name': args.model,
        'tag'       : tag,
        'state_dict': model.state_dict(),
        'optimizer_state': base_optimizer.state_dict(),
        'history'   : history,
        'val_acc'   : val_acc,
    }, ckpt_dir / "last_epoch.pth")

    # ── Save training history ─────────────────────────────────────────────────
    with open(metr_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f" Training complete. Best val acc: {best_val_acc*100:.2f}% at epoch {best_epoch}")
    print(f"{'='*70}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # POST-TRAINING: Load best model and do full evaluation + feature extraction
    # ─────────────────────────────────────────────────────────────────────────
    print("📊 Loading best model for final evaluation & feature extraction...")
    best_ckpt = torch.load(ckpt_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt['state_dict'])
    model.eval()

    # ── Extract & save features: BOTH train and test splits ──────────────────
    print("\n📦 Saving train-split features (needed for t-SNE, calibration-set selection)...")
    _, _, _, _ = extract_and_save_features(
        model, train_loader, device, feat_dir, split='train'
    )

    print("\n📦 Saving test-split features...")
    test_probs, test_preds, test_labels, test_meta_df = extract_and_save_features(
        model, test_loader, device, feat_dir, split='test'
    )

    # ── Full metric suite on test split ──────────────────────────────────────
    print("\n📐 Computing full metric suite on test split...")
    metrics = compute_all_metrics(
        probs   = test_probs,
        preds   = test_preds,
        labels  = test_labels,
        meta_df = test_meta_df,
    )

    print("\n" + "─" * 60)
    print(f"  Accuracy          : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1          : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1       : {metrics['weighted_f1']:.4f}")
    print(f"  Macro PR-AUC      : {metrics['macro_pr_auc']:.4f}")
    print(f"  ECE               : {metrics['ece']:.4f}")
    print("\n  Per-Class PR-AUC:")
    for cls, auc in metrics['pr_auc_per_class'].items():
        print(f"    {cls:20s}: {auc:.4f}")
    print("\n  Per-Operator Accuracy:")
    for op, acc in metrics.get('per_operator_accuracy', {}).items():
        print(f"    {op:20s}: {acc:.4f}")
    print("\n  Per-Machine Accuracy:")
    for mach, acc in metrics.get('per_machine_accuracy', {}).items():
        print(f"    {mach:20s}: {acc:.4f}")
    print("\n  Classification Report:")
    print(metrics['classification_report'])
    print("─" * 60)

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    save_metrics_json(metrics, metr_dir / "all_metrics.json")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n🎨 Generating plots...")
    title = f"{args.model} [{tag}]"

    plot_training_curves(
        history, plot_dir / "training_curves.png", title=f"Training Curves — {title}"
    )
    plot_confusion_matrix(
        test_labels, test_preds,
        plot_dir / "confusion_matrix.png", title=f"Confusion Matrix — {title}"
    )
    plot_per_class_metrics(
        metrics, plot_dir / "per_class_metrics.png", title=f"Per-Class Metrics — {title}"
    )
    plot_reliability_diagram(
        test_probs, test_labels,
        plot_dir / "reliability_diagram.png", title=f"Reliability Diagram — {title}"
    )
    plot_per_group_accuracy(
        metrics, plot_dir / "per_group_accuracy.png", title=f"Per-Group Accuracy — {title}"
    )

    # ── W&B: log final metrics and plots ─────────────────────────────────────
    if use_wandb:
        try:
            import wandb
            wandb.log({
                'final/accuracy'    : metrics['accuracy'],
                'final/macro_f1'    : metrics['macro_f1'],
                'final/macro_pr_auc': metrics['macro_pr_auc'],
                'final/ece'         : metrics['ece'],
            })
            for p in (plot_dir / "confusion_matrix.png",
                      plot_dir / "per_class_metrics.png",
                      plot_dir / "reliability_diagram.png",
                      plot_dir / "training_curves.png",
                      plot_dir / "per_group_accuracy.png"):
                if p.exists():
                    wandb.log({p.stem: wandb.Image(str(p))})
            wandb.finish()
        except Exception as e:
            print(f"[W&B] Failed to log final metrics: {e}")

    print(f"\n✅ All done! Outputs in: {run_dir}")
    print(f"   ├── checkpoints/ (best_model.pth, last_epoch.pth, epoch_*.pth)")
    print(f"   ├── features/    (train/test embeddings, probs, labels, meta CSVs)")
    print(f"   ├── metrics/     (all_metrics.json, training_history.json)")
    print(f"   └── plots/       (all PNG plots)")


if __name__ == "__main__":
    main()
