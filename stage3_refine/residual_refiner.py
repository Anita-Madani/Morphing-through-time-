"""
Stage 3 — Residual flow refinement.

A lightweight U-Net that takes the two endpoint images (00.png, 04.png) and the
composed RoMa flow (Stage 2) and predicts a residual correction:

    refined_flow = roma_flow + residual

Dataset layout (one folder per sample):
    <root>/<sample>/00.png          endpoint image A
    <root>/<sample>/04.png          endpoint image B
    <root>/<sample>/roma_flow.npy   composed RoMa flow, HxWx2   (from Stage 2)
    <root>/<sample>/gt_flow.npy     ground-truth flow, 2xHxW    (affine GT)

Train:
    python residual_refiner.py --data <root> --epochs 20 --ckpt_dir checkpoints_residual
"""
import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

H, W = 256, 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- model
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResidualRefinerNet(nn.Module):
    """U-Net encoder over concat(img0, img4), fused with the RoMa flow at the
    bottleneck, decoding a residual added back to the input flow."""

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = conv_block(6, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.enc5 = conv_block(512, 1024)
        self.enc6 = conv_block(1024, 2048)

        # RoMa flow fusion at the bottleneck
        self.flow_proj = nn.Conv2d(2, 2048, kernel_size=1)
        self.se = SEBlock(2048)

        # Decoder
        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec5 = conv_block(2048 + 1024, 1024)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec4 = conv_block(1024 + 512, 512)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = conv_block(512 + 256, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = conv_block(256 + 128, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = conv_block(128 + 64, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
        )

    def forward(self, img0, img4, roma_flow):
        x = torch.cat([img0, img4], dim=1)          # [B, 6, H, W]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e6 = self.enc6(self.pool(e5))               # [B, 2048, H/32, W/32]

        roma_f = F.interpolate(roma_flow, size=e6.shape[-2:], mode="bilinear", align_corners=True)
        fused = self.se(e6 + self.flow_proj(roma_f))

        d5 = self.dec5(torch.cat([self.up5(fused), e5], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        residual = self.final(d1)
        residual = F.interpolate(residual, size=roma_flow.shape[-2:], mode="bilinear", align_corners=True)
        return roma_flow + residual


# ---------------------------------------------------------------- dataset
class ResidualFlowDataset(Dataset):
    def __init__(self, root_dir, split="train", split_file=None, save_split=False,
                 val_ratio=0.2, test_ratio=0.2):
        self.root_dir = root_dir
        self.to_tensor = transforms.ToTensor()
        self.split = split

        # keep only complete samples
        self.samples = sorted(
            s for s in os.listdir(root_dir)
            if os.path.exists(os.path.join(root_dir, s, "00.png"))
            and os.path.exists(os.path.join(root_dir, s, "04.png"))
            and os.path.exists(os.path.join(root_dir, s, "roma_flow.npy"))
            and os.path.exists(os.path.join(root_dir, s, "gt_flow.npy"))
        )

        if split_file:
            with open(os.path.join(split_file, f"{split}.txt")) as f:
                self.samples = [line.strip() for line in f if line.strip()]
        elif save_split:
            random.seed(42)
            random.shuffle(self.samples)
            n = len(self.samples)
            n_test = int(n * test_ratio)
            n_val = int((n - n_test) * val_ratio)
            test_s = self.samples[:n_test]
            val_s = self.samples[n_test:n_test + n_val]
            train_s = self.samples[n_test + n_val:]
            os.makedirs("splits", exist_ok=True)
            for name, s in [("train", train_s), ("val", val_s), ("test", test_s)]:
                with open(f"splits/{name}.txt", "w") as f:
                    f.writelines(x + "\n" for x in s)
            self.samples = {"train": train_s, "val": val_s, "test": test_s}[split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        base = os.path.join(self.root_dir, sample)
        img0 = self.to_tensor(Image.open(os.path.join(base, "00.png")).convert("RGB"))
        img4 = self.to_tensor(Image.open(os.path.join(base, "04.png")).convert("RGB"))
        roma_flow = torch.from_numpy(np.load(os.path.join(base, "roma_flow.npy"))).permute(2, 0, 1).float()
        gt_flow = torch.from_numpy(np.load(os.path.join(base, "gt_flow.npy"))).float()
        return img0, img4, roma_flow, gt_flow, os.path.basename(base)


# ---------------------------------------------------------------- training
def train(model, train_loader, val_loader, ckpt_dir, epochs=20, lr=1e-4, resume=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    start_epoch = 0

    if resume:
        print(f"Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for img0, img4, roma_flow, gt_flow, _ in train_loader:
            img0, img4 = img0.to(DEVICE), img4.to(DEVICE)
            roma_flow, gt_flow = roma_flow.to(DEVICE), gt_flow.to(DEVICE)
            pred_flow = model(img0, img4, roma_flow)
            loss = loss_fn(pred_flow, gt_flow)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img0, img4, roma_flow, gt_flow, _ in val_loader:
                img0, img4 = img0.to(DEVICE), img4.to(DEVICE)
                roma_flow, gt_flow = roma_flow.to(DEVICE), gt_flow.to(DEVICE)
                val_loss += loss_fn(model(img0, img4, roma_flow), gt_flow).item()
        avg_val_loss = val_loss / max(1, len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs} | train {avg_loss:.4f} | val {avg_val_loss:.4f}")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, os.path.join(ckpt_dir, f"flow_epoch_{epoch + 1}.pth"))

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset root (one folder per sample).")
    parser.add_argument("--ckpt_dir", default="checkpoints_residual", help="Where to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    train_ds = ResidualFlowDataset(root_dir=args.data, split="train", save_split=True)
    val_ds = ResidualFlowDataset(root_dir=args.data, split="val", split_file="splits")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    train(ResidualRefinerNet(), train_loader, val_loader,
          ckpt_dir=args.ckpt_dir, epochs=args.epochs, lr=args.lr, resume=args.resume)
