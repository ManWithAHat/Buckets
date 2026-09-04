"""
Simple bucket classifier using CLIP embeddings + logistic regression.

Usage:
    1. Put your images in:
         data/bucket/       (10 images containing your bucket)
         data/no_bucket/    (10 images without it)
    2. Run this script: python bucket_classifier.py
    3. To classify a new image from your own code:
         from bucket_classifier import classify_bucket
         label, confidence = classify_bucket("path/to/new_image.jpg")
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from transformers import CLIPModel, CLIPProcessor

DATA_DIR = Path("data")
BUCKET_DIR = DATA_DIR / "bucket"
NO_BUCKET_DIR = DATA_DIR / "no_bucket"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_NAME = "openai/clip-vit-base-patch32"

clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()

_classifier = None  # trained lazily / cached after first fit


def list_images(folder: Path):
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS)


@torch.no_grad()
def embed_image(image_path) -> np.ndarray:
    """Return a normalized CLIP embedding vector for one image."""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    feats = clip_model.get_image_features(**inputs)

    # Some transformers versions return a wrapped output object here instead
    # of a plain tensor — unwrap it if needed.
    if not torch.is_tensor(feats):
        feats = getattr(feats, "image_embeds", None) or feats.pooler_output

    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]


def train_classifier(bucket_dir: Path = BUCKET_DIR, no_bucket_dir: Path = NO_BUCKET_DIR):
    """Train (and cache) a logistic regression classifier on your labeled images."""
    global _classifier

    bucket_images = list_images(bucket_dir)
    no_bucket_images = list_images(no_bucket_dir)
    if not bucket_images or not no_bucket_images:
        raise FileNotFoundError(
            f"Expected images in '{bucket_dir}' and '{no_bucket_dir}'. "
            f"Found {len(bucket_images)} and {len(no_bucket_images)}."
        )

    X = np.array([embed_image(p) for p in bucket_images + no_bucket_images])
    y = np.array([1] * len(bucket_images) + [0] * len(no_bucket_images))

    # Leave-one-out cross-validation to sanity-check accuracy on this small dataset
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X[test_idx])
    acc = accuracy_score(y, preds)
    print(f"Leave-one-out accuracy: {acc:.2f} ({(y == preds).sum()}/{len(y)} correct)")

    # Final classifier trained on all available data
    _classifier = LogisticRegression(max_iter=1000).fit(X, y)
    return _classifier


def classify_bucket(image_path):
    """Return ('bucket' | 'no_bucket', confidence) for a single image."""
    global _classifier
    if _classifier is None:
        train_classifier()

    emb = embed_image(image_path).reshape(1, -1)
    pred = _classifier.predict(emb)[0]
    confidence = _classifier.predict_proba(emb)[0][int(pred)]
    label = "bucket" if pred == 1 else "no_bucket"
    return label, float(confidence)


if __name__ == "__main__":
    train_classifier()

    # Quick demo: classify everything in data/bucket and data/no_bucket
    for folder in (BUCKET_DIR, NO_BUCKET_DIR):
        for img_path in list_images(folder):
            label, confidence = classify_bucket(img_path)
            print(f"{img_path.name:30s} -> {label:10s} ({confidence:.2f})")
