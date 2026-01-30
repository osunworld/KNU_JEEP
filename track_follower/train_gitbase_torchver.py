import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ----------------------------
# Config
# ----------------------------
DATA_PATH = "/abr/coss11/repo/robot_data"
MODEL_OUT = "/abr/coss11/repo/Track_Model_pytorch.pt"

BATCH_SIZE = 32
EPOCHS = 1000
LR = 1e-3
VAL_RATIO = 0.1
PATIENCE = 20
MIN_DELTA = 1e-4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ----------------------------
# Dataset
# ----------------------------
class TrackDataset(Dataset):
    def __init__(self, data_path):
        self.image_files = glob.glob(os.path.join(data_path, "track_dataset_*", "*.jpg"))
        self.image_files.sort()
        print(f"Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file = self.image_files[idx]
        filename = os.path.basename(file)

        # label parse: "x_y_....jpg" or "x_y.jpg"
        parts = filename.split("_")
        x = float(parts[0])
        y = float(parts[1].split(".")[0])  # y가 바로 확장자면
        label = np.array([x, y], dtype=np.float32) / 400.0

        img = cv2.imread(file)  # BGR
        if img is None:
            # 만약 깨진 파일이면 0 반환 (학습은 진행되지만 가능하면 데이터 정리 추천)
            img = np.zeros((270, 400, 3), dtype=np.uint8)

        # crop: [120:270] -> H=150
        img = img[120:270, :, :]  # (150, W, 3)

        # normalize to [0,1]
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))  # (3,150,400)

        return torch.from_numpy(img), torch.from_numpy(label)

# ----------------------------
# Model (Res-like CNN regressor)
# ----------------------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(x + out)

class TrackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResBlock(32),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            ResBlock(64),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            ResBlock(128),
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2),  # 회귀 출력 2개 (x,y)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = self.head(x)
        return x

# ----------------------------
# Train / Eval
# ----------------------------
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            n += bs
    return total_loss / max(n, 1)

def main():
    ds = TrackDataset(DATA_PATH)
    if len(ds) == 0:
        raise RuntimeError("No images found. DATA_PATH / 패턴 확인해줘.")

    val_len = int(len(ds) * VAL_RATIO)
    train_len = len(ds) - val_len

    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = TrackNet().to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    bad = 0

    print("Starting training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

            bs = x.size(0)
            total += loss.item() * bs
            n += bs

        train_loss = total / max(n, 1)
        val_loss = evaluate(model, val_loader, loss_fn)

        print(f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # Early Stopping
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "img_shape": (3, 150, 400),
                },
                MODEL_OUT,
            )
            print(f"  ✅ saved best -> {MODEL_OUT}")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch} (best_val={best_val:.6f})")
                break

    print("Done.")

if __name__ == "__main__":
    main()