import os
import glob
import random
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================
# Config
# =========================
DATA_ROOT = "."
POS_DIRS = [os.path.join(DATA_ROOT, "green_hit_image1")]  # label=1
NEG_DIRS = [
    os.path.join(DATA_ROOT, "frames_green_only"),     # label=0
    os.path.join(DATA_ROOT, "frames_no_green"),       # label=0
]

IMG_W, IMG_H = 50, 75     # (W,H) => 최종 입력은 (3,75,50)
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
SEED = 42

SAVE_PATH = "green_ped_model_3.pt"   # torch weights
THRESH = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)


# =========================
# Dataset
# =========================
class GreenPedDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples  # list of (path, label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)  # BGR
        if img is None:
            # 아주 드물게 깨진 파일이면 검은 이미지로 대체
            img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        x = torch.tensor(img, dtype=torch.float32)
        y = torch.tensor([label], dtype=torch.float32)  # (1,)
        return x, y


# =========================
# Model (가볍고 정확하게)
# =========================
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 75x50 -> 37x25

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 37x25 -> 18x12

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> 64x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 1),  # logits
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits  # BCEWithLogitsLoss 사용


# =========================
# Split + Train
# =========================
def make_samples():
    pos_files = []
    for d in POS_DIRS:
        pos_files += list_images(d)

    neg_files = []
    for d in NEG_DIRS:
        neg_files += list_images(d)

    samples = [(p, 1) for p in pos_files] + [(p, 0) for p in neg_files]
    random.shuffle(samples)

    print(f"[DATA] pos={len(pos_files)}, neg={len(neg_files)}, total={len(samples)}")
    return samples, len(pos_files), len(neg_files)


def train():
    set_seed(SEED)

    samples, npos, nneg = make_samples()
    if len(samples) == 0:
        raise RuntimeError("No images found. Check folder paths.")

    # 90/10 split
    split = int(0.9 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_ds = GreenPedDataset(train_samples)
    val_ds = GreenPedDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TinyCNN().to(DEVICE)

    # 클래스 불균형 보정(1:300, 0:800)
    # pos_weight = (neg/pos) 형태로 주면 pos(1) 실수 더 강하게 벌줌
    pos_weight = torch.tensor([nneg / max(npos, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    patience = 8
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * x.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= THRESH).float()
            tr_correct += (preds == y).sum().item()
            tr_total += y.numel()

        tr_loss /= max(tr_total, 1)
        tr_acc = tr_correct / max(tr_total, 1)

        # ---- val ----
        model.eval()
        va_loss = 0.0
        va_correct = 0
        va_total = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                logits = model(x)
                loss = criterion(logits, y)

                va_loss += loss.item() * x.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= THRESH).float()
                va_correct += (preds == y).sum().item()
                va_total += y.numel()

        va_loss /= max(va_total, 1)
        va_acc = va_correct / max(va_total, 1)

        print(f"[E{epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        # early stop + best save
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "img_w": IMG_W,
                    "img_h": IMG_H,
                    "thresh": THRESH,
                },
                SAVE_PATH,
            )
            print(f"✅ saved best model -> {SAVE_PATH} (val_acc={best_val_acc:.4f})")
        else:
            bad += 1
            if bad >= patience:
                print("⛔ Early stop")
                break

    print("Done.")


if __name__ == "__main__":
    train()