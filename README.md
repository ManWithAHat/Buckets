# Bucket Classifier

Classifies whether an image contains a bucket, using CLIP embeddings +
logistic regression. Trains on a small labeled set (10 images with a
bucket, 10 without) and works even for unusual/specific bucket types since
it learns from your own examples rather than a generic text prompt.

## Setup

```bash
pip install torch transformers scikit-learn pillow --break-system-packages
```

## Data layout

```
data/
  bucket/        # images that contain the bucket
  no_bucket/     # images that don't
```

## Usage

Train and run the demo classification over your labeled folders:

```bash
python bucket_classifier.py
```

Use it in your own code:

```python
from bucket_classifier import classify_bucket

label, confidence = classify_bucket("path/to/new_image.jpg")
print(label, confidence)
```
