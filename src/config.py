"""
Central configuration for all experiments.
All paths, hyperparameters, and flags live here.
"""
import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(r"r:\HTIC")
DATA_DIR      = ROOT / "FETAL_PLANES_ZENODO"
IMG_DIR       = DATA_DIR / "Images"
CSV_PATH      = DATA_DIR / "FETAL_PLANES_DB_data.csv"
OUTPUT_DIR    = ROOT / "outputs"
SRC_DIR       = ROOT / "src"

# ─── Dataset ──────────────────────────────────────────────────────────────────
IMG_SIZE      = 224          # resize all images to this
MEAN          = [0.485, 0.456, 0.406]   # ImageNet stats (grayscale → 3ch repeat)
STD           = [0.229, 0.224, 0.225]

# 6 coarse classes (official Plane column exact strings)
CLASSES_6 = [
    "Fetal abdomen",
    "Fetal brain",
    "Fetal femur",
    "Fetal thorax",
    "Maternal cervix",
    "Other",
]
NUM_CLASSES = 6

# 3 fine-grain brain sub-classes  (only used for analysis, not training target)
BRAIN_SUBCLASSES = ["Trans-thalamic", "Trans-cerebellum", "Trans-ventricular"]

CLASS_TO_IDX = {c: i for i, c in enumerate(sorted(CLASSES_6))}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
NUM_WORKERS     = 0          # 0 = safest on Windows CPU-only (no spawn issues)
PIN_MEMORY      = False      # No GPU, keep False
EPOCHS          = 50
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1
PATIENCE        = 10         # Early stopping patience (epochs)
TORCH_THREADS   = 8          # OMP/MKL threads; uses all Ryzen 7730U cores

# ASAM hyperparams
ASAM_RHO   = 0.5
ASAM_ETA   = 0.01

# CutMix / Mixup (from timm)
MIXUP_ALPHA    = 0.8
CUTMIX_ALPHA   = 1.0
MIXUP_PROB     = 0.5         # probability of applying either mixup or cutmix

# Checkpoint: save every N epochs (in addition to best + last)
SAVE_EVERY_N   = 10

# ─── Models to train & analyze ──────────────────────────────────────────────
# Implemented and evaluated model: EfficientNet-B0 with AdamW
MODELS = [
    {
        "name"      : "efficientnet_b0",
        "timm_name" : "efficientnet_b0",
        "use_asam"  : False,
        "use_cutmix": False,
        "tag"       : "adamw",
    },
]

# ─── Weights & Biases ─────────────────────────────────────────────────────────
USE_WANDB      = False       # set True if you have wandb configured
WANDB_PROJECT  = "HTIC-fetal-planes"
WANDB_ENTITY   = None        # set to your wandb username if needed

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ─── Feature saving (for t-SNE, UMAP, calibration) ───────────────────────────
SAVE_FEATURES  = True        # always True — no "I should have saved" moments

# ─── Test-run (sanity check with 2 epochs + small data) ──────────────────────
TEST_RUN_EPOCHS = 2
TEST_RUN_SUBSET = 200        # images per split during test run
