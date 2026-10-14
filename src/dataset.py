"""
Dataset & DataLoader for FETAL_PLANES_DB.
Handles:
  - CSV parsing with semicolon separator
  - Official train/test split (Train column)
  - Grayscale PNG → 3-channel RGB (repeat)
  - Class-weighted sampling
  - Returns full metadata per sample for later per-operator / per-machine analysis
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from config import (
    IMG_DIR, CSV_PATH, IMG_SIZE, MEAN, STD,
    CLASS_TO_IDX, IDX_TO_CLASS, CLASSES_6,
    NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    MIXUP_ALPHA, CUTMIX_ALPHA, MIXUP_PROB, SEED,
    TEST_RUN_SUBSET
)


# ─────────────────────────────────────────────────────────────────────────────
def load_dataframe(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load CSV, clean columns, add label_idx."""
    df = pd.read_csv(csv_path, sep=';')
    # Strip whitespace from all column names and string values
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # Map Plane → label index (only 6-class coarse label used for training)
    df['label'] = df['Plane'].map(CLASS_TO_IDX)

    # Sanity: drop rows where label is NaN (shouldn't happen)
    bad = df['label'].isna().sum()
    if bad > 0:
        print(f"[WARNING] Dropping {bad} rows with unknown Plane label")
        df = df.dropna(subset=['label'])

    df['label'] = df['label'].astype(int)
    df['img_path'] = df['Image_name'].apply(lambda x: str(IMG_DIR / (x + '.png')))

    print(f"[Dataset] Loaded {len(df)} rows")
    print(f"[Dataset] Train: {df['Train'].sum()} | Test: {(df['Train']==0).sum()}")
    print(f"[Dataset] Class distribution:\n{df['Plane'].value_counts()}\n")
    return df


def get_transforms(split: str) -> T.Compose:
    """
    split: 'train' or 'val'/'test'
    Grayscale images are loaded as RGB (L→RGB via PIL) then normalized.
    """
    if split == 'train':
        return T.Compose([
            T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),    # slightly bigger then crop
            T.RandomCrop(IMG_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])
    else:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])


class FetalPlanesDataset(Dataset):
    """
    PyTorch Dataset for FETAL_PLANES_DB.
    Each item returns:
        image  : Tensor [3, H, W]
        label  : int (0-5)
        meta   : dict with image_name, patient_num, operator, machine, brain_plane
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['img_path']

        # Load as RGB (handles grayscale PNGs transparently)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = int(row['label'])

        meta = {
            'image_name' : row['Image_name'],
            'patient_num': int(row['Patient_num']),
            'operator'   : str(row['Operator']),
            'machine'    : str(row['US_Machine']),
            'brain_plane': str(row['Brain_plane']),
            'plane'      : str(row['Plane']),
        }

        return img, label, meta


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Class-weighted sampler to handle imbalance.
    Each class gets weight = 1 / class_count, assigned per sample.
    """
    class_counts = df['label'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[df['label'].values]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def get_dataloaders(
    test_run: bool = False,
    use_weighted_sampler: bool = True,
):
    """
    Returns (train_loader, test_loader, train_df, test_df, class_weights_tensor)
    """
    df = load_dataframe()

    train_df = df[df['Train'] == 1].copy()
    test_df  = df[df['Train'] == 0].copy()

    if test_run:
        # Stratified subset for smoke test
        n_train_per = max(1, TEST_RUN_SUBSET // NUM_CLASSES)
        n_test_per  = max(1, TEST_RUN_SUBSET // NUM_CLASSES // 2)
        train_df = pd.concat([
            grp.sample(min(len(grp), n_train_per), random_state=SEED)
            for _, grp in train_df.groupby('label')
        ]).reset_index(drop=True)
        test_df = pd.concat([
            grp.sample(min(len(grp), n_test_per), random_state=SEED)
            for _, grp in test_df.groupby('label')
        ]).reset_index(drop=True)
        print(f"[TestRun] Reduced train={len(train_df)}, test={len(test_df)}")

    train_dataset = FetalPlanesDataset(train_df, transform=get_transforms('train'))
    test_dataset  = FetalPlanesDataset(test_df,  transform=get_transforms('test'))

    # Class weights for loss function (inverse frequency)
    counts = train_df['label'].value_counts().sort_index().values.astype(np.float32)
    class_weights = torch.tensor(1.0 / counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES  # normalize

    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_df)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            collate_fn=meta_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            collate_fn=meta_collate_fn,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=meta_collate_fn,
    )

    print(f"[DataLoader] Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    return train_loader, test_loader, train_df, test_df, class_weights


def meta_collate_fn(batch):
    """
    Custom collate to handle the meta dict in each sample.
    Returns: images, labels, list_of_meta_dicts
    """
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    metas  = [b[2] for b in batch]
    return images, labels, metas


if __name__ == "__main__":
    train_loader, test_loader, train_df, test_df, class_weights = get_dataloaders(test_run=False)
    print("Class weights:", class_weights)
    imgs, lbls, metas = next(iter(train_loader))
    print("Batch image shape:", imgs.shape)
    print("Batch labels:", lbls)
    print("Sample meta:", metas[0])
