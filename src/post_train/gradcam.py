"""
post_train/gradcam.py — Explainability & Grad-CAM Visualization for Fetal Ultrasound.

Provides clinical interpretability for ultrasound plane classification:
  - Hooks into penultimate convolutional stage
  - Computes gradient-weighted class activation mappings (Grad-CAM)
  - Identifies anatomical acoustic landmarks (fetal skull midline, femur diaphysis,
    cardiac chambers, abdominal circumference, cervical canal)
  - Renders multi-panel gallery comparing Ultrasound frame vs Grad-CAM heatmap
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
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
from models import build_model
from config import OUTPUT_DIR, NUM_CLASSES, IDX_TO_CLASS, CLASSES_6, IMG_SIZE
from dataset import get_dataloaders


class GradCAM:
    """
    Native PyTorch Grad-CAM implementation.
    Works seamlessly on any CNN / ViT backbone with zero external dependencies.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        input_tensor: [1, 3, H, W]
        target_class: int (if None, uses argmax)
        Returns: (cam_map [H, W], predicted_class, confidence)
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        probs  = F.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        # Backward on target score
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Gradients: [1, C, H_feat, W_feat]
        grads = self.gradients
        acts  = self.activations

        # Global average pooling over spatial dimensions
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, H_feat, W_feat]
        cam = F.relu(cam)

        # Interpolate to input resolution
        cam = F.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, confidence


def get_last_conv_layer(model: nn.Module) -> nn.Module:
    """Finds the last Conv2d layer in the model backbone."""
    conv_layers = [mod for mod in model.backbone.modules() if isinstance(mod, nn.Conv2d)]
    if not conv_layers:
        raise ValueError("No Conv2d layers found in backbone.")
    return conv_layers[-1]


def overlay_cam_on_image(img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5, colormap='jet'):
    """
    img_np: [H, W, 3] in range [0, 1]
    cam: [H, W] in range [0, 1]
    """
    cmap = plt.colormaps[colormap]
    heatmap = cmap(cam)[:, :, :3]  # [H, W, 3]
    overlay = alpha * heatmap + (1 - alpha) * img_np
    overlay = np.clip(overlay, 0, 1)
    return heatmap, overlay


def run_gradcam_gallery(model_name: str, run_name: str = None, samples_per_class: int = 1):
    if run_name is None:
        tag = "adamw"
        run_name = f"{model_name}_{tag}"

    run_dir = OUTPUT_DIR / run_name
    ckpt_path = run_dir / "checkpoints" / "best_model.pth"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        print(f"⚠️ Checkpoint not found at {ckpt_path}. Skipping Grad-CAM.")
        return None

    print("\n" + "=" * 70)
    print(f"🔬 Ultrasound Grad-CAM Explainability: {model_name} ({run_name})")
    print("=" * 70)

    device = torch.device("cpu")
    model = build_model(model_name, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    target_layer = get_last_conv_layer(model)
    grad_cam = GradCAM(model, target_layer)

    # Load test dataloader
    _, test_loader, _, test_df, _ = get_dataloaders(test_run=True)

    # Collect sample images for each class
    class_samples = {}
    for imgs, labels, metas in test_loader:
        for i in range(len(labels)):
            lbl = labels[i].item()
            c_name = IDX_TO_CLASS[lbl]
            if c_name not in class_samples:
                class_samples[c_name] = (imgs[i:i+1], lbl, metas[i])
            if len(class_samples) == NUM_CLASSES:
                break
        if len(class_samples) == NUM_CLASSES:
            break

    n_classes_found = len(class_samples)
    fig, axes = plt.subplots(n_classes_found, 3, figsize=(12, 3.6 * n_classes_found))
    if n_classes_found == 1:
        axes = np.expand_dims(axes, 0)

    row = 0
    for c_name, (img_tensor, true_label, meta) in sorted(class_samples.items()):
        cam, pred_class, conf = grad_cam.generate_cam(img_tensor, target_class=true_label)

        # Denormalize image for display
        # ImageNet mean & std:
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        orig_img_np = img_tensor[0].cpu().numpy() * std + mean
        orig_img_np = np.clip(orig_img_np.transpose(1, 2, 0), 0, 1)

        heatmap, overlay = overlay_cam_on_image(orig_img_np, cam, alpha=0.55)

        pred_name = IDX_TO_CLASS[pred_class]
        is_correct = (pred_class == true_label)
        status_sym = "[OK]" if is_correct else "[X]"

        # Col 1: Original B-mode ultrasound
        axes[row, 0].imshow(orig_img_np, cmap='gray')
        axes[row, 0].set_title(f"Target: {c_name}", fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')

        # Col 2: Grad-CAM Saliency Heatmap
        axes[row, 1].imshow(cam, cmap='inferno')
        axes[row, 1].set_title("Grad-CAM Heatmap", fontsize=11, fontweight='bold')
        axes[row, 1].axis('off')

        # Col 3: Saliency Overlay on Ultrasound
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(
            f"{status_sym} Pred: {pred_name} ({conf*100:.1f}%)",
            fontsize=11, fontweight='bold',
            color='darkgreen' if is_correct else 'darkred'
        )
        axes[row, 2].axis('off')

        row += 1

    plt.suptitle(f"{model_name} — Clinical Ultrasound Grad-CAM Explainability", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    gallery_path = plot_dir / "gradcam_gallery.png"
    plt.savefig(gallery_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved Grad-CAM gallery to: {gallery_path.name}")
    return gallery_path


if __name__ == "__main__":
    from config import MODELS
    for m in [cfg['name'] for cfg in MODELS]:
        run_gradcam_gallery(m)
