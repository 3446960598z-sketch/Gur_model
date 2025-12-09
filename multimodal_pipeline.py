#!/usr/bin/env python3
"""
Multi-modal feature extraction and lightweight training for Xiaohongshu content.

Requirements (install if missing):
    pip install torch torchvision transformers pandas requests pillow

Usage:
    python multimodal_pipeline.py --csv contents.csv

The script will:
- load and clean rows (needs title & desc and at least one image/video)
- extract text embeddings via bert-base-chinese
- extract image/video frame embeddings via ResNet50
- fuse features and train a small MLP to predict liked_count
- save fused features to features.csv and the model to model.pth
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from typing import List

import pandas as pd
import requests
import torch
from PIL import Image
from torch import nn
from torchvision import models
from transformers import AutoModel, AutoTokenizer

CACHE_DIR = "./cache"
TEXT_DIM = 768
IMG_DIM = 2048
VID_DIM = 2048
FUSED_DIM = TEXT_DIM + IMG_DIM + VID_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="Xiaohongshu multimodal feature extractor")
    parser.add_argument("--csv", default="contents.csv", help="Path to raw csv data")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    return parser.parse_args()


def ensure_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)


def clean_row(row) -> bool:
    # Require title and desc
    if pd.isna(row.get("title")) or pd.isna(row.get("desc")):
        return False
    has_text = str(row["title"]).strip() or str(row["desc"]).strip()
    # Need at least one image or video
    has_video = bool(str(row.get("video_url") or "").strip())
    has_image = len(parse_image_list(row.get("image_list"))) > 0
    return bool(has_text and (has_video or has_image))


def parse_image_list(val) -> List[str]:
    if val is None:
        return []
    text = str(val).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data if str(x).startswith("http")]
    except Exception:
        pass
    # fallback: split by common separators
    parts = [p.strip().strip("'\" ") for p in re_split(text)]
    return [p for p in parts if p.startswith("http")]


def re_split(text: str) -> List[str]:
    for sep in ["|", ",", " ", "\t", "\n"]:
        if sep in text:
            return text.split(sep)
    return [text]


def download_file(url: str, prefix: str) -> str:
    try:
        fname = os.path.join(CACHE_DIR, f"{prefix}_{os.path.basename(url).split('?')[0]}")
        if not os.path.exists(fname):
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            with open(fname, "wb") as f:
                f.write(resp.content)
        return fname
    except Exception as e:
        print(f"[warn] failed to download {url}: {e}")
        return ""


def load_text_model(device):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese").to(device)
    model.eval()
    return tokenizer, model


def get_text_embedding(title: str, desc: str, tokenizer, model, device):
    text = (title or "") + " [SEP] " + (desc or "")
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs).last_hidden_state[:, 0]
    return output.squeeze(0).cpu()


def load_image_backbone(device):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    base = models.resnet50(weights=weights)
    backbone = nn.Sequential(*list(base.children())[:-1]).to(device)
    backbone.eval()
    preprocess = weights.transforms()
    return backbone, preprocess


def image_to_embedding(path: str, backbone, preprocess, device):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = backbone(tensor).flatten(1)
        return feat.squeeze(0).cpu()
    except Exception as e:
        print(f"[warn] image embedding failed for {path}: {e}")
        return torch.zeros(IMG_DIM)


def extract_video_frame(video_url: str, prefix: str) -> str:
    if not shutil.which("ffmpeg"):
        return ""
    video_path = download_file(video_url, prefix)
    if not video_path:
        return ""
    frame_path = os.path.join(CACHE_DIR, f"{prefix}_frame.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vframes",
        "1",
        frame_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return frame_path
    except Exception as e:
        print(f"[warn] ffmpeg failed for {video_url}: {e}")
        return ""


def fuse_features(text_emb, image_emb, video_emb):
    parts = []
    parts.append(text_emb if text_emb is not None else torch.zeros(TEXT_DIM))
    parts.append(image_emb if image_emb is not None else torch.zeros(IMG_DIM))
    parts.append(video_emb if video_emb is not None else torch.zeros(VID_DIM))
    return torch.cat(parts)


def build_model(input_dim=FUSED_DIM):
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
    )


def main():
    args = parse_args()
    ensure_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.csv)
    df = df[df.apply(clean_row, axis=1)].reset_index(drop=True)
    if df.empty:
        print("No valid data after cleaning.")
        sys.exit(0)

    tokenizer, text_model = load_text_model(device)
    img_backbone, img_pre = load_image_backbone(device)

    fused_list = []
    feature_records = []

    for idx, row in df.iterrows():
        note_id = row.get("note_id", f"idx{idx}")
        title = str(row.get("title", ""))
        desc = str(row.get("desc", ""))
        text_emb = get_text_embedding(title, desc, tokenizer, text_model, device)

        image_emb = None
        images = parse_image_list(row.get("image_list"))
        if images:
            img_path = download_file(images[0], f"{note_id}_img")
            if img_path:
                image_emb = image_to_embedding(img_path, img_backbone, img_pre, device)

        video_emb = None
        video_url = row.get("video_url")
        if isinstance(video_url, str) and video_url.strip():
            frame_path = extract_video_frame(video_url, f"{note_id}_vid")
            if frame_path:
                video_emb = image_to_embedding(frame_path, img_backbone, img_pre, device)

        fused = fuse_features(text_emb, image_emb, video_emb)
        fused_list.append(fused)
        feature_records.append(
            {
                "note_id": note_id,
                "liked_count": row.get("liked_count", 0),
                "text_emb": json.dumps(text_emb.tolist()),
                "image_emb": json.dumps(image_emb.tolist() if image_emb is not None else []),
                "video_emb": json.dumps(video_emb.tolist() if video_emb is not None else []),
                "fused_emb": json.dumps(fused.tolist()),
            }
        )

    # Save features
    feat_df = pd.DataFrame(feature_records)
    feat_df.to_csv("features.csv", index=False)
    print(f"Saved features for {len(feat_df)} samples to features.csv")

    features = torch.stack(fused_list).float()
    targets = torch.tensor([float(x) for x in df["liked_count"].fillna(0).tolist()]).float().unsqueeze(1)

    # Train/val split
    indices = list(range(len(features)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    if not train_idx:
        train_idx = indices
    if not val_idx:
        val_idx = train_idx
    train_x, val_x = features[train_idx], features[val_idx]
    train_y, val_y = targets[train_idx], targets[val_idx]

    model = build_model(FUSED_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        preds = model(train_x.to(device))
        loss = criterion(preds, train_y.to(device))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(val_x.to(device))
            val_loss = criterion(val_preds, val_y.to(device))
        print(f"Epoch {epoch}: train_loss={loss.item():.4f} val_loss={val_loss.item():.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")


if __name__ == "__main__":
    main()
