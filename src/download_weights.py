"""
Automated Weight & Feature Embeddings Downloader
HTIC IIT Madras Assignment: EfficientNet-B0 Optimization & Edge Deployment

This script downloads the trained PyTorch checkpoint (best_model.pth, 48.7 MB)
and test embeddings (test_embeddings.npy, 27 MB) from GitHub Releases or Google Drive.
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

# Default Release URL (GitHub Releases allows up to 2GB files with chunked uploads)
GITHUB_REPO = "rashiniyasp/Deep-Learning-Model-Optimization-using-EfficientNet-B0"
RELEASE_TAG = "v1.0.0"

CHECKPOINT_DIR = Path("outputs/efficientnet_b0_adamw/checkpoints")
FEATURES_DIR = Path("outputs/efficientnet_b0_adamw/features")

FILES = {
    "best_model.pth": {
        "url": f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/best_model.pth",
        "dest": CHECKPOINT_DIR / "best_model.pth",
        "expected_size_mb": 48.65,
        "description": "Trained EfficientNet-B0 PyTorch Checkpoint (94.67% Acc)"
    },
    "test_embeddings.npy": {
        "url": f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/test_embeddings.npy",
        "dest": FEATURES_DIR / "test_embeddings.npy",
        "expected_size_mb": 26.98,
        "description": "Extracted Penultimate Feature Embeddings (5,271 x 1280)"
    }
}


def download_with_progress(url: str, dest_path: Path, desc: str):
    """Download a file with an interactive CLI progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    print(f"\n📥 Downloading: {desc}")
    print(f"   Source : {url}")
    print(f"   Target : {dest_path}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            bar_len = 30
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stdout.write(f"\r   [{bar}] {percent:5.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, temp_path, reporthook=reporthook)
        print("\n   ✓ Download complete.")
        if temp_path.exists():
            temp_path.replace(dest_path)
        return True
    except Exception as e:
        print(f"\n   ❌ Download failed: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(description="Download model weights and feature embeddings.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist.")
    parser.add_argument("--gdrive_id", type=str, default=None, help="Optional Google Drive File ID for best_model.pth.")
    args = parser.parse_args()

    print("=" * 70)
    print("  Artifact Downloader: EfficientNet-B0 Optimization")
    print(f"  Repository: https://github.com/{GITHUB_REPO}")
    print("=" * 70)

    for filename, info in FILES.items():
        dest = info["dest"]
        if dest.exists() and not args.force:
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"\n✓ Found existing {filename} ({size_mb:.2f} MB) at {dest}")
            continue

        success = download_with_progress(info["url"], dest, info["description"])
        if not success:
            print("\n💡 NOTE FOR REPO OWNER:")
            print(f"   Please create a GitHub Release '{RELEASE_TAG}' at:")
            print(f"   https://github.com/{GITHUB_REPO}/releases/new")
            print(f"   and attach '{filename}' to the release assets.")
            print("   Or provide a public Google Drive link.")


if __name__ == "__main__":
    main()
