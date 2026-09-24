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

class BucketClosedError(Exception):
    """Raised when the bucket's lid is closed, so fullness cannot be determined."""
    pass

PROJECT_DIR = Path(__file__).resolve().parent

BUCKET_DIR = PROJECT_DIR / "bucket"
NO_BUCKET_DIR = PROJECT_DIR / "no_bucket"
FULL_DIR = PROJECT_DIR / "full"
NOT_FULL_DIR = PROJECT_DIR / "not_full"
CLOSED_DIR = PROJECT_DIR / "closed"
OPEN_DIR = PROJECT_DIR / "open"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_NAME = "openai/clip-vit-base-patch32"

clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()

_classifier = None  # trained lazily / cached after first fit
_fullness_classifier = None
_openness_classifier = None

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

def train_fullness_classifier(full_dir: Path = FULL_DIR, not_full_dir: Path = NOT_FULL_DIR):
    """Train (and cache) a full/not_full classifier. Only call on images that contain a bucket."""
    global _fullness_classifier

    full_images = list_images(full_dir)
    not_full_images = list_images(not_full_dir)
    if not full_images or not not_full_images:
        raise FileNotFoundError(
            f"Expected images in '{full_dir}' and '{not_full_dir}'. "
            f"Found {len(full_images)} and {len(not_full_images)}."
        )

    X = np.array([embed_image(p) for p in full_images + not_full_images])
    y = np.array([1] * len(full_images) + [0] * len(not_full_images))

    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X[test_idx])
    acc = accuracy_score(y, preds)
    print(f"[fullness] Leave-one-out accuracy: {acc:.2f} ({(y == preds).sum()}/{len(y)} correct)")

    _fullness_classifier = LogisticRegression(max_iter=1000).fit(X, y)
    return _fullness_classifier

def train_openness_classifier(closed_dir: Path = CLOSED_DIR, open_dir: Path = OPEN_DIR):
    """Train (and cache) a closed/open classifier. Only call on images that contain a bucket."""
    global _openness_classifier

    closed_images = list_images(closed_dir)
    open_images = list_images(open_dir)
    if not closed_images or not open_images:
        raise FileNotFoundError(
            f"Expected images in '{closed_dir}' and '{open_dir}'. "
            f"Found {len(closed_images)} and {len(open_images)}."
        )

    X = np.array([embed_image(p) for p in closed_images + open_images])
    y = np.array([1] * len(closed_images) + [0] * len(open_images))

    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X[test_idx])
    acc = accuracy_score(y, preds)
    print(f"[openness] Leave-one-out accuracy: {acc:.2f} ({(y == preds).sum()}/{len(y)} correct)")

    _openness_classifier = LogisticRegression(max_iter=1000).fit(X, y)
    return _openness_classifier


def classify_openness(image_path):
    """Return ('closed' | 'open', confidence). Assumes the image already contains a bucket."""
    global _openness_classifier
    if _openness_classifier is None:
        train_openness_classifier()

    emb = embed_image(image_path).reshape(1, -1)
    pred = _openness_classifier.predict(emb)[0]
    confidence = _openness_classifier.predict_proba(emb)[0][int(pred)]
    label = "closed" if pred == 1 else "open"
    return label, float(confidence)

def classify_fullness(image_path):
    """Return ('full' | 'not_full', confidence). Assumes the image already contains a bucket."""
    global _fullness_classifier
    if _fullness_classifier is None:
        train_fullness_classifier()

    emb = embed_image(image_path).reshape(1, -1)
    pred = _fullness_classifier.predict(emb)[0]
    confidence = _fullness_classifier.predict_proba(emb)[0][int(pred)]
    label = "full" if pred == 1 else "not_full"
    return label, float(confidence)

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

def classify_bucket_hierarchical(image_path):
    """
    Full pipeline: bucket/no_bucket, then (if bucket) full/not_full.
    """
    bucket_label, bucket_conf = classify_bucket(image_path)

    if bucket_label == "no_bucket":
        return {
            "bucket": bucket_label,
            "bucket_confidence": bucket_conf,
            "fullness": None,
            "fullness_confidence": None,
        }

    openness_label, openness_conf = classify_openness(image_path)
    if openness_label == "closed":
        raise BucketClosedError(
            f"Bucket detected (confidence {bucket_conf:.2f}) but lid is closed "
            f"(confidence {openness_conf:.2f}) — cannot determine fullness."
        )

    fullness_label, fullness_conf = classify_fullness(image_path)
    return {
        "bucket": bucket_label,
        "bucket_confidence": bucket_conf,
        "openness": openness_label,
        "openness_confidence": openness_conf,
        "fullness": fullness_label,
        "fullness_confidence": fullness_conf,
    }

if __name__ == "__main__":
    train_classifier()
    train_fullness_classifier()
    train_openness_classifier()

    for folder in (BUCKET_DIR, NO_BUCKET_DIR):
        for img_path in list_images(folder):
            try:
                result = classify_bucket_hierarchical(img_path)
                print(f"{img_path.name:30s} -> {result}")
            except BucketClosedError as e:
                print(f"{img_path.name:30s} -> ERROR: {e}")
