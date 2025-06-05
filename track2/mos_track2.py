#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from audiobox_aesthetics.model.aes import AesMultiOutput, Normalize
from loss import preference_aware_pamr_loss

AXES_NAME = ["CE", "CU", "PC", "PQ"]

# ---- 1. Dataset：16kHz & 單聲道、10s 隨機截取、RMS 正規化、標準化 label ----
class AesCSVDataset(Dataset):
    def __init__(self, df, sample_rate=16000, chunk_sec=10, target_mean=None, target_std=None):
        """
        df: pandas.DataFrame，需含 data_path 與四軸分數欄位
        target_mean/std: torch.Tensor shape=[4]，訓練集分數的均值與標準差
        """
        self.df = df
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_sec * sample_rate
        self.score_cols = ["Content_Enjoyment","Content_Usefulness",
                           "Production_Complexity","Production_Quality"]
        assert target_mean is not None and target_std is not None, "請提供標準化參數"
        self.target_mean = target_mean.to(torch.float32)
        self.target_std  = target_std.to(torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        wav, sr = torchaudio.load(row["data_path"])
        # mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # resample
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # random 10s crop or pad
        T = wav.shape[-1]
        if T > self.chunk_samples:
            start = random.randint(0, T - self.chunk_samples)
            wav = wav[..., start:start + self.chunk_samples]
            mask = torch.ones(self.chunk_samples, dtype=torch.bool)
        else:
            pad_len = self.chunk_samples - T
            wav = F.pad(wav, (0, pad_len))
            # 前 T frames 有效、後 pad_len frames 無效
            mask = torch.cat([
                torch.ones(T, dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool)
                ], dim=0)

        # RMS loudness normalization
        rms = torch.sqrt(torch.mean(wav**2) + 1e-9)
        wav = wav / rms

        target = torch.tensor([row[c] for c in self.score_cols], dtype=torch.float32)

        # label standardization
        target = (target - self.target_mean) / (self.target_std + 1e-9)

        # wav: [1, chunk_samples], mask: [chunk_samples], target: [4]
        return wav, mask.unsqueeze(0), target

# ---- 2. collate_fn ----
def collate_fn(batch):
    wavs, masks, targets = zip(*batch)
    wavs   = torch.stack(wavs,   dim=0)  # [B,1,chunk_samples]
    masks  = torch.stack(masks,  dim=0)  # [B,1,chunk_samples]
    targets = torch.stack(targets, dim=0) # [B,4]
    return {"wav": wavs, "mask": masks}, targets

# ---- 3. Loss：MSE + MAE ----
def combined_loss(preds, targets):
    return F.mse_loss(preds, targets) + F.l1_loss(preds, targets)

# ---- 4. Training & Validation Loop ----
def train_epoch(model, loader, optimizer, device, args, target_mean, target_std):
    model.train()
    total_ce = total_cu = total_pc = total_pq = 0.0
    total_samples = 0
    for batch, targets in tqdm(loader, desc="Train Progress"):
        x = {
            "wav":  batch["wav"].to(device),
            "mask": batch["mask"].to(device),
        }
        preds_dict = model(x)
        preds = torch.stack([preds_dict[k] for k in AXES_NAME], dim=1).float() # 確保 preds 是 float
        targets = targets.to(device)

        # 反向標準化 preds 和 targets
        preds_original = torch.zeros_like(preds, device=device)
        targets_original = torch.zeros_like(targets, device=device)
        for i in range(len(AXES_NAME)):
            preds_original[:, i] = preds[:, i] * target_std[i] + target_mean[i]
            targets_original[:, i] = targets[:, i] * target_std[i] + target_mean[i]

        loss_ce = combined_loss(preds[:,0], targets[:,0]) + args.ranking_loss_weight * preference_aware_pamr_loss(preds_original[:,0], targets_original[:,0], task='ce', preference_factor=args.preference_factor)
        loss_cu = combined_loss(preds[:,1], targets[:,1]) + args.ranking_loss_weight * preference_aware_pamr_loss(preds_original[:,1], targets_original[:,1], task='cu', preference_factor=args.preference_factor)
        loss_pc = combined_loss(preds[:,2], targets[:,2]) + args.ranking_loss_weight * preference_aware_pamr_loss(preds_original[:,2], targets_original[:,2], task='pc', preference_factor=args.preference_factor)
        loss_pq = combined_loss(preds[:,3], targets[:,3]) + args.ranking_loss_weight * preference_aware_pamr_loss(preds_original[:,3], targets_original[:,3], task='pq', preference_factor=args.preference_factor)

        # 四個 loss 相加為總 loss
        loss = loss_ce + loss_cu + loss_pc + loss_pq

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bsz = preds.size(0)
        total_ce += loss_ce.item() * bsz
        total_cu += loss_cu.item() * bsz
        total_pc += loss_pc.item() * bsz
        total_pq += loss_pq.item() * bsz
        total_samples += bsz

    return {
        "ce": total_ce / total_samples,
        "cu": total_cu / total_samples,
        "pc": total_pc / total_samples,
        "pq": total_pq / total_samples,
        "total": (total_ce + total_cu + total_pc + total_pq) / total_samples,
    }

def eval_epoch(model, loader, device, args, target_mean, target_std):
    model.eval()
    total_ce = total_cu = total_pc = total_pq = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch, targets in tqdm(loader, desc="Validating Progress"):
            x = {
                "wav":  batch["wav"].to(device),
                "mask": batch["mask"].to(device),
            }
            preds_dict = model(x)
            preds = torch.stack([preds_dict[k] for k in AXES_NAME], dim=1).float() # 確保 preds 是 float
            targets = targets.to(device)

            # 反向標準化 preds 和 targets
            preds_original = torch.zeros_like(preds, device=device)
            targets_original = torch.zeros_like(targets, device=device)
            for i in range(len(AXES_NAME)):
                preds_original[:, i] = preds[:, i] * target_std[i] + target_mean[i]
                targets_original[:, i] = targets[:, i] * target_std[i] + target_mean[i]

            loss_ce = combined_loss(preds[:,0], targets[:,0]) + args.ranking_loss_weight * preference_aware_pamr_loss(preds_original[:,0], targets_original[:,0], task='ce', preference_factor=args.preference_factor)
            loss_cu = combined_loss(preds[:,1], targets[:,1]) + args.ranking_loss_weight * preference_aware_pamr_loss(preds_original[:,1], targets_original[:,1], task='cu', preference_factor=args.preference_factor)
            loss_pc = combined_loss(preds[:,2], targets[:,2]) + args.ranking_loss_weight * preference_aware_pamr_loss(preds_original[:,2], targets_original[:,2], task='pc', preference_factor=args.preference_factor)
            loss_pq = combined_loss(preds[:,3], targets[:,3]) + args.ranking_loss_weight * preference_aware_pamr_loss(preds_original[:,3], targets_original[:,3], task='pq', preference_factor=args.preference_factor)

            bsz = preds.size(0)
            total_ce += loss_ce.item() * bsz
            total_cu += loss_cu.item() * bsz
            total_pc += loss_pc.item() * bsz
            total_pq += loss_pq.item() * bsz
            total_samples += bsz

    return {
        "ce": total_ce / total_samples,
        "cu": total_cu / total_samples,
        "pc": total_pc / total_samples,
        "pq": total_pq / total_samples,
        "total": (total_ce + total_cu + total_pc + total_pq) / total_samples,
    }

# ---- 5. Main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",  type=str,   default="../audiomos2025_track2/audiomos2025-track2-train_list_filter.csv", required=False)
    parser.add_argument("--dev_csv",    type=str,   default="../audiomos2025_track2/audiomos2025-track2-dev_list_filter.csv", required=False)
    parser.add_argument("--exp_name",   type=str,   default="./exp")
    parser.add_argument("--epochs",     type=int,   default=1000)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size",  type=int, default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--seed",       type=int,   default=1992,   help="隨機種子，用於 reproducibility")
    parser.add_argument("--patience",   type=int,   default=20,     help="Early stopping patience")
    parser.add_argument("--freeze_encoder", type=lambda x: bool(int(x)), default=1, help="是否凍結 WavLM encoder (1=凍結, 0=不凍結)")
    parser.add_argument("--ranking_loss_weight",         type=float, default=1.0)
    parser.add_argument("--margin_scale",         type=float, default=0.2)
    parser.add_argument("--preference_factor",         type=float, default=7.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.exp_name, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    # 讀 CSV
    train_df = pd.read_csv(args.train_csv)
    dev_df   = pd.read_csv(args.dev_csv)

    # 載入模型
    model = AesMultiOutput.from_pretrained("facebook/audiobox-aesthetics", freeze_encoder=bool(args.freeze_encoder))
    model.to(device)

    # 從 model.target_transform 中擷取官方 mean/std
    # 注意順序要與 score_cols 一致：["Content_Enjoyment","Content_Usefulness","Production_Complexity","Production_Quality"]
    stats_mean = []
    stats_std  = []
    for axis in AXES_NAME:
        stats_mean.append(model.target_transform[axis]["mean"])
        stats_std.append(model.target_transform[axis]["std"])
    target_mean = torch.tensor(stats_mean, dtype=torch.float32)
    target_std  = torch.tensor(stats_std,  dtype=torch.float32)

    # Dataset & DataLoader
    train_ds = AesCSVDataset(train_df, target_mean=target_mean, target_std=target_std)
    dev_ds   = AesCSVDataset(dev_df,   target_mean=target_mean, target_std=target_std)
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=8)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.eval_batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 優化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    best_dev = float("inf")
    best_file = None
    epochs_no_improve = 0
    target_mean, target_std = target_mean.to(device), target_std.to(device)

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_losses = train_epoch(model, train_loader, optimizer, device, args, target_mean, target_std)
        dev_losses   = eval_epoch(model, dev_loader, device, args, target_mean, target_std)

        # 取出 total loss
        train_total = train_losses["total"]
        dev_total   = dev_losses["total"]

        # 印出
        print(
            f"[Epoch {epoch}] "
            f"train_ce={train_losses['ce']:.4f}, "
            f"train_cu={train_losses['cu']:.4f}, "
            f"train_pc={train_losses['pc']:.4f}, "
            f"train_pq={train_losses['pq']:.4f} || "
            f"dev_ce={dev_losses['ce']:.4f}, "
            f"dev_cu={dev_losses['cu']:.4f}, "
            f"dev_pc={dev_losses['pc']:.4f}, "
            f"dev_pq={dev_losses['pq']:.4f}"
        )

        if dev_total < best_dev:
            best_dev = dev_total
            epochs_no_improve = 0
            ckpt_file = os.path.join(args.exp_name, f"best_model_{epoch}.pt")
            if best_file is not None and os.path.exists(best_file):
                os.remove(best_file)
            best_file = ckpt_file
            torch.save({"state_dict": model.state_dict()}, ckpt_file)
            print(f"👉 Saved best model: {ckpt_file}")
        else:
            epochs_no_improve += 1
            print(f'patience: {args.patience - epochs_no_improve}')
            if epochs_no_improve >= args.patience:
                print(f"🔹 Early stopping after {args.patience} epochs without improvement.")
                break

    # 同步儲存 HuggingFace 格式
    model.save_pretrained(args.exp_name)
    print(f"✅ 訓練完成，模型與標準化參數已儲存到 {args.exp_name}")

if __name__ == "__main__":
    main()
