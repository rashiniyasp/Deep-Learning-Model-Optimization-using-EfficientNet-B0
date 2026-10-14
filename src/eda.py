"""
EDA — Exploratory Data Analysis for FETAL_PLANES_DB.
Run once before training to understand the data.

python src/eda.py

Saves all plots to outputs/eda/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from tqdm import tqdm

from config import (
    CSV_PATH, IMG_DIR, OUTPUT_DIR, IDX_TO_CLASS, CLASS_TO_IDX, CLASSES_6
)

EDA_DIR = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11})


def load_df():
    df = pd.read_csv(CSV_PATH, sep=';')
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    df['split'] = df['Train'].map({1: 'Train', 0: 'Test'})
    return df


def plot_class_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Overall
    counts = df['Plane'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor='black', lw=0.8)
    axes[0].set_title("Class Distribution (All Data)")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=30)
    for i, (cls, cnt) in enumerate(counts.items()):
        axes[0].text(i, cnt + 20, str(cnt), ha='center', va='bottom', fontsize=10)

    # Train vs Test per class
    ct = df.groupby(['Plane', 'split']).size().unstack(fill_value=0)
    ct.plot(kind='bar', ax=axes[1], color=['steelblue', 'darkorange'], edgecolor='black', lw=0.5)
    axes[1].set_title("Class Distribution — Train vs Test")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend(title="Split")

    plt.suptitle("FETAL_PLANES_DB — Class Distribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(EDA_DIR / "class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] class_distribution.png saved")


def plot_brain_subclasses(df):
    brain = df[df['Plane'] == 'Fetal brain']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = brain['Brain_plane'].value_counts()
    axes[0].bar(counts.index, counts.values, color=plt.cm.Pastel1(np.linspace(0, 1, len(counts))),
                edgecolor='black', lw=0.8)
    axes[0].set_title("Brain Sub-class Distribution (All)")
    axes[0].set_xlabel("Brain Sub-class")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=20)

    ct = brain.groupby(['Brain_plane', 'split']).size().unstack(fill_value=0)
    ct.plot(kind='bar', ax=axes[1], color=['steelblue', 'darkorange'], edgecolor='black', lw=0.5)
    axes[1].set_title("Brain Sub-class — Train vs Test")
    axes[1].set_xlabel("Brain Sub-class")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis='x', rotation=20)
    axes[1].legend(title="Split")

    plt.suptitle("Brain Sub-class Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(EDA_DIR / "brain_subclasses.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] brain_subclasses.png saved")


def plot_metadata(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    op_counts = df['Operator'].value_counts()
    axes[0].bar(op_counts.index, op_counts.values,
                color=plt.cm.Set2(np.linspace(0, 1, len(op_counts))), edgecolor='black', lw=0.5)
    axes[0].set_title("Images per Operator")
    axes[0].set_xlabel("Operator")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=30)

    mach_counts = df['US_Machine'].value_counts()
    axes[1].bar(mach_counts.index, mach_counts.values,
                color=plt.cm.Set1(np.linspace(0, 1, len(mach_counts))), edgecolor='black', lw=0.5)
    axes[1].set_title("Images per US Machine")
    axes[1].set_xlabel("US Machine")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis='x', rotation=30)

    plt.suptitle("Metadata Distribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(EDA_DIR / "metadata_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] metadata_distribution.png saved")


def plot_sample_images(df, n_per_class=4):
    """Show sample images from each of the 6 classes."""
    classes = sorted(df['Plane'].unique())
    n_cls   = len(classes)
    fig, axes = plt.subplots(n_cls, n_per_class, figsize=(n_per_class * 3, n_cls * 3))

    for row, cls in enumerate(classes):
        cls_df = df[df['Plane'] == cls].sample(n=min(n_per_class, len(df[df['Plane'] == cls])),
                                                random_state=42)
        for col, (_, sample) in enumerate(cls_df.iterrows()):
            img_path = IMG_DIR / (sample['Image_name'] + '.png')
            img = Image.open(img_path).convert('RGB')
            ax  = axes[row][col]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(cls, fontsize=10, fontweight='bold', rotation=0,
                              labelpad=60, va='center')

    plt.suptitle("Sample Images per Class", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(EDA_DIR / "sample_images.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] sample_images.png saved")


def analyze_image_sizes(df, n_sample=500):
    """Sample images to check if all sizes are uniform."""
    sample = df.sample(n=min(n_sample, len(df)), random_state=42)
    widths, heights = [], []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Checking image sizes"):
        try:
            img = Image.open(IMG_DIR / (row['Image_name'] + '.png'))
            widths.append(img.width)
            heights.append(img.height)
        except Exception:
            pass

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(widths,  bins=30, color='steelblue', edgecolor='black', lw=0.5)
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (px)")

    axes[1].hist(heights, bins=30, color='darkorange', edgecolor='black', lw=0.5)
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height (px)")

    print(f"  Width  : min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
    print(f"  Height : min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")

    plt.suptitle(f"Image Size Distribution (n={len(widths)} sampled)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(EDA_DIR / "image_sizes.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] image_sizes.png saved")


def main():
    print("=" * 60)
    print("  EDA — FETAL_PLANES_DB")
    print("=" * 60)

    df = load_df()

    print(f"\nTotal images : {len(df)}")
    print(f"Train        : {(df['Train']==1).sum()}")
    print(f"Test         : {(df['Train']==0).sum()}")
    print(f"Patients     : {df['Patient_num'].nunique()}")
    print(f"Operators    : {df['Operator'].nunique()} — {df['Operator'].unique().tolist()}")
    print(f"US Machines  : {df['US_Machine'].nunique()} — {df['US_Machine'].unique().tolist()}")
    print(f"\nClass distribution:\n{df['Plane'].value_counts()}")
    print(f"\nBrain sub-classes:\n{df[df['Plane']=='Fetal brain']['Brain_plane'].value_counts()}")

    # Save summary CSV
    summary = df.groupby(['Plane', 'split']).size().reset_index(name='count')
    summary.to_csv(EDA_DIR / "class_summary.csv", index=False)

    print("\n[EDA] Generating EDA plots...")
    plot_class_distribution(df)
    plot_brain_subclasses(df)
    plot_metadata(df)
    plot_sample_images(df, n_per_class=4)
    analyze_image_sizes(df, n_sample=300)

    print(f"\n[EDA] Completed successfully. Saved to: {EDA_DIR}")


if __name__ == "__main__":
    main()
