"""
post_train/tsne_umap.py — Latent Space Manifold Analysis (t-SNE & UMAP).

Projects penultimate 384D / 1280D / 2048D embeddings into 2D:
  1. Anatomical Plane Separation (6 coarse classes)
  2. Clinical Operator Robustness (Junior vs Senior sonographers)
  3. Ultrasound Device Invariance (Machine model domain shift)
  4. Fine-grained Fetal Brain Sub-plane Separation (Trans-thalamic, Trans-cerebellum, Trans-ventricular)
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
from config import OUTPUT_DIR, CLASSES_6, IDX_TO_CLASS


def run_manifold_analysis(model_name: str, run_name: str = None, perplexity: int = 30):
    if run_name is None:
        tag = "adamw"
        run_name = f"{model_name}_{tag}"

    run_dir = OUTPUT_DIR / run_name
    feat_dir = run_dir / "features"
    plot_dir = run_dir / "plots"
    for d in [feat_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    emb_path = feat_dir / "test_embeddings.npy"
    meta_path = feat_dir / "test_meta.csv"

    if not emb_path.exists() or not meta_path.exists():
        print(f"⚠️ Features not found in {feat_dir}. Skipping manifold analysis.")
        return None

    print("\n" + "=" * 70)
    print(f"🌐 Latent Space Manifold Analysis (t-SNE & UMAP): {model_name} ({run_name})")
    print("=" * 70)

    embeddings = np.load(emb_path)
    meta_df = pd.read_csv(meta_path)
    n_samples = len(embeddings)
    print(f"  Loaded {n_samples} embeddings of dimension {embeddings.shape[1]}")

    # Adjust perplexity for small test runs
    eff_perp = min(perplexity, max(5, n_samples // 4))

    # ── 1. t-SNE Projection ──────────────────────────────────────────────────
    print(f"  Computing t-SNE (perplexity={eff_perp})...")
    tsne = TSNE(n_components=2, perplexity=eff_perp, random_state=42, max_iter=1000)
    tsne_2d = tsne.fit_transform(embeddings)

    # ── 2. UMAP Projection ───────────────────────────────────────────────────
    if HAS_UMAP and n_samples >= 15:
        print("  Computing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, n_samples - 2))
        umap_2d = reducer.fit_transform(embeddings)
    else:
        umap_2d = None

    # Save 2D coordinates for interactive inspection
    coords_dict = {'tsne_2d': tsne_2d}
    if umap_2d is not None:
        coords_dict['umap_2d'] = umap_2d
    np.savez(feat_dir / "2d_projections.npz", **coords_dict)

    # Add coordinates to DataFrame for plotting
    plot_df = meta_df.copy()
    plot_df['tsne_x'] = tsne_2d[:, 0]
    plot_df['tsne_y'] = tsne_2d[:, 1]
    if umap_2d is not None:
        plot_df['umap_x'] = umap_2d[:, 0]
        plot_df['umap_y'] = umap_2d[:, 1]

    # Map class index to string
    plot_df['class_name'] = plot_df['label'].map(IDX_TO_CLASS)

    # ── Plot 1: 6 Anatomical Planes ──────────────────────────────────────────
    plot_projection(
        plot_df, x_col='tsne_x', y_col='tsne_y', hue_col='class_name',
        title=f'{model_name} — t-SNE Latent Space: 6 Fetal Anatomical Planes',
        save_path=plot_dir / "tsne_anatomical_planes.png"
    )
    if umap_2d is not None:
        plot_projection(
            plot_df, x_col='umap_x', y_col='umap_y', hue_col='class_name',
            title=f'{model_name} — UMAP Latent Space: 6 Fetal Anatomical Planes',
            save_path=plot_dir / "umap_anatomical_planes.png"
        )

    # ── Plot 2: Clinical Operator Experience (Junior vs Senior) ─────────────
    if 'Operator' in plot_df.columns and plot_df['Operator'].nunique() > 1:
        plot_projection(
            plot_df, x_col='tsne_x', y_col='tsne_y', hue_col='Operator',
            title=f'{model_name} — Feature Robustness across Operator Experience',
            save_path=plot_dir / "tsne_operator_robustness.png",
            palette={'Junior': '#f59e0b', 'Senior': '#3b82f6', 'Other': '#9ca3af'}
        )

    # ── Plot 3: Ultrasound Device Invariance ─────────────────────────────────
    if 'Device' in plot_df.columns and plot_df['Device'].nunique() > 1:
        plot_projection(
            plot_df, x_col='tsne_x', y_col='tsne_y', hue_col='Device',
            title=f'{model_name} — Domain Invariance across Ultrasound Devices',
            save_path=plot_dir / "tsne_device_invariance.png"
        )

    # ── Plot 4: Fetal Brain Sub-planes (Fine-grained) ─────────────────────────
    brain_df = plot_df[plot_df['class_name'] == 'Fetal brain'].copy()
    if len(brain_df) > 5 and 'Brain_plane' in brain_df.columns:
        brain_subclasses = brain_df[brain_df['Brain_plane'].isin(
            ['Trans-thalamic', 'Trans-cerebellum', 'Trans-ventricular']
        )]
        if len(brain_subclasses) > 5:
            plot_projection(
                brain_subclasses, x_col='tsne_x', y_col='tsne_y', hue_col='Brain_plane',
                title=f'{model_name} — Fine-Grained Brain Sub-plane Clustering',
                save_path=plot_dir / "tsne_brain_subclasses.png"
            )

    print(f"  ✅ All manifold plots saved to: {plot_dir}")
    return coords_dict


def plot_projection(df, x_col, y_col, hue_col, title, save_path, palette=None):
    plt.figure(figsize=(9, 7))
    sns.set_theme(style='whitegrid')

    # Draw scatter
    if palette:
        ax = sns.scatterplot(
            data=df, x=x_col, y=y_col, hue=hue_col,
            palette=palette, s=55, alpha=0.85, edgecolor='black', linewidth=0.5
        )
    else:
        ax = sns.scatterplot(
            data=df, x=x_col, y=y_col, hue=hue_col,
            s=55, alpha=0.85, edgecolor='black', linewidth=0.5
        )

    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Component 1', fontsize=11, fontweight='bold')
    plt.ylabel('Component 2', fontsize=11, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved plot: {save_path.name}")


if __name__ == "__main__":
    from config import MODELS
    for m in [cfg['name'] for cfg in MODELS]:
        run_manifold_analysis(m)
