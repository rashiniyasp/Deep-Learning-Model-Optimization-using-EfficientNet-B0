"""
Feature extraction & saving utilities.
Saves everything needed for:
  - t-SNE / UMAP visualization  (features + labels + meta)
  - Calibration-set selection   (K-Means on train embeddings)
  - Grad-CAM                    (weights already in checkpoint)
  - Confusion matrix            (predictions CSV)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F


def extract_and_save_features(
    model,
    loader,
    device: torch.device,
    save_dir: Path,
    split: str,          # 'train' or 'test'
):
    """
    Run full forward pass, collect:
      - embeddings    [N, embed_dim]   .npy
      - logits        [N, num_classes] .npy
      - probs         [N, num_classes] .npy  (softmax)
      - preds         [N]              .npy
      - labels        [N]              .npy
      - meta.csv      one row per image

    Returns: (probs_np, preds_np, labels_np, meta_df)
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    all_embeddings = []
    all_logits     = []
    all_probs      = []
    all_preds      = []
    all_labels     = []
    all_metas      = []

    with torch.no_grad():
        for imgs, labels, metas in tqdm(loader, desc=f"Extracting [{split}]"):
            imgs   = imgs.to(device)
            logits, feats = model.get_features(imgs)
            probs  = F.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            all_embeddings.append(feats.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_metas.extend(metas)

    embeddings_np = np.concatenate(all_embeddings, axis=0)
    logits_np     = np.concatenate(all_logits,     axis=0)
    probs_np      = np.concatenate(all_probs,      axis=0)
    preds_np      = np.concatenate(all_preds,      axis=0)
    labels_np     = np.concatenate(all_labels,     axis=0)

    meta_df = pd.DataFrame(all_metas)
    meta_df['pred']  = preds_np
    meta_df['label'] = labels_np
    # Add softmax prob columns for each class
    from config import IDX_TO_CLASS
    for c_idx, c_name in IDX_TO_CLASS.items():
        meta_df[f'prob_{c_name.replace(" ", "_")}'] = probs_np[:, c_idx]

    # ── Save ────────────────────────────────────────────────────────────────
    np.save(save_dir / f'{split}_embeddings.npy', embeddings_np)
    np.save(save_dir / f'{split}_logits.npy',     logits_np)
    np.save(save_dir / f'{split}_probs.npy',      probs_np)
    np.save(save_dir / f'{split}_preds.npy',      preds_np)
    np.save(save_dir / f'{split}_labels.npy',     labels_np)
    meta_df.to_csv(save_dir / f'{split}_meta.csv', index=False)

    print(f"[Features] {split}: embeddings{embeddings_np.shape} | "
          f"probs{probs_np.shape} | saved to {save_dir}")

    return probs_np, preds_np, labels_np, meta_df
