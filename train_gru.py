import os
import re
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def clean_text(text: str) -> str:
    """Basic text cleaning: keep Chinese characters, letters, digits and normalize spaces."""
    if not isinstance(text, str):
        text = ""
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Character-level tokenization for Chinese text."""
    return list(text)


def build_vocab(texts: List[List[str]], vocab_size: int = 8000) -> Tuple[dict, dict]:
    """Build vocabulary mapping from token to index and inverse mapping."""
    from collections import Counter

    counter = Counter()
    for tokens in texts:
        counter.update(tokens)

    most_common = counter.most_common(vocab_size - 2)  # reserve PAD=0, UNK=1
    token2idx = {token: idx + 2 for idx, (token, _) in enumerate(most_common)}
    token2idx["<PAD>"] = 0
    token2idx["<UNK>"] = 1
    idx2token = {idx: token for token, idx in token2idx.items()}
    return token2idx, idx2token


def encode_tokens(tokens: List[str], token2idx: dict, max_len: int = 50) -> List[int]:
    indices = [token2idx.get(t, token2idx["<UNK>"]) for t in tokens]
    if len(indices) < max_len:
        indices.extend([token2idx["<PAD>"]] * (max_len - len(indices)))
    else:
        indices = indices[:max_len]
    return indices


class PostDataset(Dataset):
    def __init__(self, texts: List[List[str]], labels: List[int], token2idx: dict, max_len: int = 50):
        self.texts = texts
        self.labels = labels
        self.token2idx = token2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids = encode_tokens(self.texts[idx], self.token2idx, self.max_len)
        label = float(self.labels[idx])
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.float)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_size: int = 128, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.gru(embedded)
        # Use last timestep from both directions
        last_output = outputs[:, -1, :]
        logits = self.fc(last_output)
        return logits.squeeze(1)


def prepare_data(csv_path: str, max_len: int = 50, vocab_size: int = 8000):
    df = pd.read_csv(csv_path)
    if "title" not in df.columns or "liked_count" not in df.columns:
        raise ValueError("CSV must contain 'title' and 'liked_count' columns")

    df["title_clean"] = df["title"].apply(clean_text)
    df["tokens"] = df["title_clean"].apply(tokenize)

    median_likes = df["liked_count"].median()
    df["label"] = (df["liked_count"] > median_likes).astype(int)

    token2idx, idx2token = build_vocab(df["tokens"].tolist(), vocab_size=vocab_size)
    dataset = PostDataset(df["tokens"].tolist(), df["label"].tolist(), token2idx, max_len=max_len)
    return dataset, token2idx, idx2token, median_likes


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.sigmoid(logits) >= 0.5
    correct = (preds == labels.bool()).sum().item()
    return correct / len(labels)


def train_model(
    csv_path: str = "data/posts.csv",
    batch_size: int = 64,
    num_epochs: int = 5,
    max_len: int = 50,
    vocab_size: int = 8000,
    lr: float = 1e-3,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset, token2idx, _, median_likes = prepare_data(csv_path, max_len=max_len, vocab_size=vocab_size)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = GRUClassifier(vocab_size=len(token2idx)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_size_current = batch_x.size(0)
            total_loss += loss.item() * batch_size_current
            total_acc += compute_accuracy(logits.detach().cpu(), batch_y.cpu()) * batch_size_current
            total_samples += batch_size_current

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                batch_size_current = batch_x.size(0)
                val_loss += loss.item() * batch_size_current
                val_acc += compute_accuracy(logits.cpu(), batch_y.cpu()) * batch_size_current
                val_samples += batch_size_current

        val_loss = val_loss / val_samples
        val_acc = val_acc / val_samples
        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_loss:.4f} Acc: {avg_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    os.makedirs("checkpoint", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "token2idx": token2idx, "median_likes": median_likes}, "checkpoint/gru_model.pth")
    print("Model saved to checkpoint/gru_model.pth")


if __name__ == "__main__":
    train_model()
