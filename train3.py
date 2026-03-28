"""
SVAMITVA Drone Image Segmentation - v3 Training Script
=======================================================
Fixes applied:
1. IGNORE_INDEX changed to 255 (no valid class suppression)
2. Median frequency class weights (proper rebalancing)
3. Fixed Dice loss (no broken ignore logic)
4. Combined loss: 40% CE + 60% Dice
5. Per-class IoU logging in validation
6. Pretrained ResNet34 encoder via segmentation_models_pytorch
7. Mixed precision training
8. ReduceLROnPlateau scheduler
9. Early stopping + best model checkpoint

Install dependencies:
    pip install segmentation-models-pytorch albumentations
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ─────────────────────────────────────────────
# CONFIG — edit these paths
# ─────────────────────────────────────────────
IMG_DIR    = "dataset/images"
MASK_DIR   = "dataset/masks"
NUM_CLASSES = 4
BATCH_SIZE  = 8
EPOCHS      = 60
LR          = 3e-4
VAL_SPLIT   = 0.15
PATIENCE    = 10          # early stopping
IGNORE_INDEX = 255        # use 255 for truly unlabeled pixels only
                          # if ALL pixels have valid labels → set to -100
SAVE_PATH   = "unet_best_v4.pth"

CLASS_NAMES = ["Background", "Class1_Blue", "Class2_Yellow", "Class3"]
# ↑ rename these to your actual class names


# ─────────────────────────────────────────────
# STEP 0: Count pixels per class (run ONCE)
# ─────────────────────────────────────────────
def count_pixels(mask_dir, num_classes=NUM_CLASSES):
    """
    Run this once before training to see your class distribution.
    Copy the printed counts into PIXEL_COUNTS below.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    mask_files = list(Path(mask_dir).glob("*.npy"))
    print(f"Counting pixels across {len(mask_files)} masks...")
    for f in mask_files:
        mask = np.load(f)
        for c in range(num_classes):
            counts[c] += int((mask == c).sum())
    print("\n========== PIXEL COUNTS ==========")
    for c in range(num_classes):
        pct = counts[c] / counts.sum() * 100
        print(f"  Class {c} ({CLASS_NAMES[c]}): {counts[c]:>12,}  ({pct:.2f}%)")
    print("===================================\n")
    return counts


# ─────────────────────────────────────────────
# STEP 1: Compute dataset mean/std (run ONCE)
# ─────────────────────────────────────────────
def compute_dataset_stats(img_dir, num_samples=500):
    img_files = list(Path(img_dir).glob("*.npy"))
    if len(img_files) > num_samples:
        img_files = random.sample(img_files, num_samples)

    means, stds = [], []
    for f in img_files:
        img = np.load(f).astype(np.float32)
        if img.shape[0] == 3:               # CHW → HWC
            img = np.transpose(img, (1, 2, 0))
        means.append(img.mean(axis=(0, 1)))
        stds.append(img.std(axis=(0, 1)))

    mean = np.array(means).mean(axis=0)
    std  = np.array(stds).mean(axis=0)
    print(f"\nDataset mean : {mean}")
    print(f"Dataset std  : {std}\n")
    return mean, std


# ─────────────────────────────────────────────
# STEP 2: Median-frequency class weights
# ─────────────────────────────────────────────
def median_freq_weights(pixel_counts, device, ignore_bg=False):
    """
    Median frequency balancing:
        weight_c = median_freq / freq_c
    Rare classes get higher weight, common classes lower weight.
    Set ignore_bg=True only if you truly don't want to learn background.
    """
    counts = np.array(pixel_counts, dtype=np.float64)
    freq   = counts / counts.sum()

    # Only compute median over classes with actual pixels
    valid_freq = freq[freq > 0]
    median     = np.median(valid_freq)

    weights = np.where(freq > 0, median / freq, 0.0)

    if ignore_bg:
        weights[0] = 0.0   # suppress background gradient

    # Normalize so weights sum to num_classes (keeps loss scale stable)
    nonzero = weights[weights > 0]
    weights  = weights / nonzero.mean()

    print("Class weights (median freq):")
    for c, (w, p) in enumerate(zip(weights, freq * 100)):
        print(f"  Class {c} ({CLASS_NAMES[c]}): weight={w:.4f}  freq={p:.2f}%")
    print()

    return torch.tensor(weights, dtype=torch.float32).to(device)


# ─────────────────────────────────────────────
# STEP 3: Dataset
# ─────────────────────────────────────────────
class GeoDataset(Dataset):
    def __init__(self, img_dir, mask_dir, mean, std, augment=True):
        self.img_files = []
        self.mask_dir  = Path(mask_dir)
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)

        for img_path in sorted(Path(img_dir).glob("*.npy")):
            idx = img_path.stem.split("_")[1]
            mask_path = self.mask_dir / f"mask_{idx}.npy"
            if mask_path.exists():
                self.img_files.append(img_path)

        print(f"✅ Paired samples found: {len(self.img_files)}")

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.4),
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=15, p=0.3,
                                   border_mode=0),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path  = self.img_files[idx]
        idx_num   = img_path.stem.split("_")[1]
        mask_path = self.mask_dir / f"mask_{idx_num}.npy"

        img  = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.int64)

        # Ensure HWC for albumentations
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))   # CHW → HWC

        # Dataset-level normalization
        img = (img - self.mean) / (self.std + 1e-6)

        # Augmentation
        if self.transform:
            aug  = self.transform(image=img.astype(np.float32),
                                  mask=mask.astype(np.uint8))
            img  = aug["image"]
            mask = aug["mask"].astype(np.int64)

        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))

        return (
            torch.tensor(img,  dtype=torch.float32),
            torch.tensor(mask, dtype=torch.long),
        )


# ─────────────────────────────────────────────
# STEP 4: Loss Functions
# ─────────────────────────────────────────────
def dice_loss(pred, target, num_classes=NUM_CLASSES):
    """
    Soft Dice loss over all classes.
    No ignore_index here — let CE handle that.
    Clamps target to valid range to avoid one_hot errors.
    """
    pred   = torch.softmax(pred, dim=1)
    target_clamped = target.clamp(0, num_classes - 1)
    target_oh = nn.functional.one_hot(target_clamped, num_classes) \
                  .permute(0, 3, 1, 2).float()   # (B, C, H, W)

    loss = 0.0
    for cls in range(num_classes):
        p = pred[:, cls]          # (B, H, W)
        t = target_oh[:, cls]     # (B, H, W)

        # Mask out IGNORE_INDEX pixels from dice computation
        if IGNORE_INDEX != -100:
            valid = (target != IGNORE_INDEX).float()
            p = p * valid
            t = t * valid

        intersection = (p * t).sum(dim=(1, 2))
        cardinality  = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        dice_score   = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        loss += (1.0 - dice_score.mean())

    return loss / num_classes


def combined_loss(pred, target, ce_criterion):
    """40% CE + 60% Dice — Dice directly optimizes IoU"""
    ce   = ce_criterion(pred, target)
    dice = dice_loss(pred, target)
    return 0.4 * ce + 0.6 * dice


# ─────────────────────────────────────────────
# STEP 5: Per-class IoU metric
# ─────────────────────────────────────────────
def compute_per_class_iou(pred_batch, target_batch, num_classes=NUM_CLASSES):
    """
    Returns list of per-class IoU values (nan if class absent in batch).
    pred_batch  : (B, H, W) argmax predictions
    target_batch: (B, H, W) ground truth
    """
    pred   = pred_batch.view(-1)
    target = target_batch.view(-1)

    # Exclude ignore index
    valid  = (target != IGNORE_INDEX)
    pred   = pred[valid]
    target = target[valid]

    ious = []
    for cls in range(num_classes):
        pred_c  = (pred == cls)
        tgt_c   = (target == cls)
        inter   = (pred_c & tgt_c).sum().item()
        union   = (pred_c | tgt_c).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(inter / union)
    return ious


def mean_iou(per_class_ious):
    """Mean IoU ignoring nan (absent classes)."""
    valid = [v for v in per_class_ious if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


# ─────────────────────────────────────────────
# STEP 6: Training Loop
# ─────────────────────────────────────────────
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*50}")
    print(f"  Device : {device}")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"{'='*50}\n")

    # ── Stats ──
    # First run: compute them automatically
    mean, std = compute_dataset_stats(IMG_DIR)
    # After first run, hardcode for speed:
    # mean = np.array([X.XX, X.XX, X.XX])
    # std  = np.array([X.XX, X.XX, X.XX])

    # ── Pixel counts → class weights ──
    # First run: count them automatically
    pixel_counts = count_pixels(MASK_DIR)
    # After first run, hardcode:
    # pixel_counts = np.array([3152129, 1395959, 485808, 1519704])

    # ── Datasets ──
    full_aug  = GeoDataset(IMG_DIR, MASK_DIR, mean, std, augment=True)
    full_val  = GeoDataset(IMG_DIR, MASK_DIR, mean, std, augment=False)

    n_total = len(full_aug)
    n_val   = max(1, int(VAL_SPLIT * n_total))
    n_train = n_total - n_val

    indices = list(range(n_total))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    train_ds = Subset(full_aug, train_idx)
    val_ds   = Subset(full_val, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}\n")

    # ── Model: pretrained ResNet34 encoder ──
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = NUM_CLASSES,
        activation      = None,          # raw logits
    ).to(device)
    print("Model: UNet + ResNet34 (ImageNet pretrained)\n")

    # ── Class weights ──
    # Set ignore_bg=True only if background is truly not a class you care about
    weights  = median_freq_weights(pixel_counts, device, ignore_bg=False)
    ce_loss  = nn.CrossEntropyLoss(
        weight       = weights,
        ignore_index = IGNORE_INDEX,   # 255 = truly unlabeled pixels
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # ── Scheduler: reduce LR when val mIoU stops improving ──
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
        min_lr=1e-6, verbose=True
    )

    # ── Mixed precision ──
    use_amp = (device == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_miou  = 0.0
    no_improve = 0

    print(f"{'Epoch':>6} | {'Loss':>8} | {'mIoU':>8} | {'LR':>10} | Per-class IoU")
    print("-" * 80)

    for epoch in range(1, EPOCHS + 1):

        # ────── TRAIN ──────
        model.train()
        total_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(imgs)
                loss    = combined_loss(outputs, masks, ce_loss)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ────── VALIDATE ──────
        model.eval()
        all_class_ious = [[] for _ in range(NUM_CLASSES)]

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                preds   = torch.argmax(outputs, dim=1).cpu()
                ious    = compute_per_class_iou(preds, masks.cpu())
                for c, iou in enumerate(ious):
                    if not np.isnan(iou):
                        all_class_ious[c].append(iou)

        per_class_mean = [
            float(np.mean(all_class_ious[c])) if all_class_ious[c] else float("nan")
            for c in range(NUM_CLASSES)
        ]
        val_miou = mean_iou(per_class_mean)

        scheduler.step(val_miou)
        current_lr = optimizer.param_groups[0]["lr"]

        # Format per-class IoU string
        pc_str = "  ".join(
            f"{CLASS_NAMES[c][:8]}={per_class_mean[c]:.3f}" if not np.isnan(per_class_mean[c])
            else f"{CLASS_NAMES[c][:8]}=  nan "
            for c in range(NUM_CLASSES)
        )

        print(f"{epoch:>6} | {avg_loss:>8.4f} | {val_miou:>8.4f} | {current_lr:>10.2e} | {pc_str}")

        # ── Save best ──
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "best_miou"  : best_miou,
                "mean"       : mean,
                "std"        : std,
                "class_names": CLASS_NAMES,
            }, SAVE_PATH)
            print(f"  ✅ Saved best model  mIoU={best_miou:.4f}")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"\n⏹ Early stopping at epoch {epoch}  (no improvement for {PATIENCE} epochs)")
            break

    print(f"\n🏁 Training complete.  Best Val mIoU: {best_miou:.4f}")
    print(f"   Model saved to: {SAVE_PATH}")


# ─────────────────────────────────────────────
# STEP 7: Inference / Visualization helper
# ─────────────────────────────────────────────
def predict_single(img_path, checkpoint_path=SAVE_PATH):
    """
    Load best model and predict a single image.
    Usage:
        pred_mask = predict_single("dataset/images/image_42.npy")
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    mean = checkpoint["mean"]
    std  = checkpoint["std"]

    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,
        in_channels     = 3,
        classes         = NUM_CLASSES,
        activation      = None,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    img = np.load(img_path).astype(np.float32)
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img = (img - mean) / (std + 1e-6)
    img = np.transpose(img, (2, 0, 1))

    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    pred = torch.argmax(output, dim=1).squeeze(0).numpy()
    return pred


def visualize_predictions(num_samples=6, checkpoint_path=SAVE_PATH):
    """Quick visual check: Satellite | Ground Truth | Prediction"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    mean = checkpoint["mean"]
    std  = checkpoint["std"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,
        in_channels     = 3,
        classes         = NUM_CLASSES,
        activation      = None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ds = GeoDataset(IMG_DIR, MASK_DIR, mean, std, augment=False)
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))

    # Color map: black=bg, blue=class1, yellow=class2, red=class3
    cmap_colors = np.array([
        [0,   0,   0  ],   # class 0 — background
        [0,   0,   255],   # class 1 — blue
        [255, 255, 0  ],   # class 2 — yellow
        [255, 0,   0  ],   # class 3 — red
    ], dtype=np.uint8)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for row, idx in enumerate(indices):
        img_tensor, mask = ds[idx]

        with torch.no_grad():
            out  = model(img_tensor.unsqueeze(0).to(device))
            pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

        # Denormalize image for display
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        img_np = (img_np * std + mean).clip(0, 255).astype(np.uint8)

        mask_np = mask.numpy()
        gt_rgb   = cmap_colors[mask_np.clip(0, NUM_CLASSES-1)]
        pred_rgb = cmap_colors[pred.clip(0, NUM_CLASSES-1)]

        axes[row][0].imshow(img_np);      axes[row][0].set_title("Satellite Image")
        axes[row][1].imshow(gt_rgb);      axes[row][1].set_title("Ground Truth")
        axes[row][2].imshow(pred_rgb);    axes[row][2].set_title("Prediction")

        for ax in axes[row]:
            ax.axis("off")

    patches = [mpatches.Patch(color=cmap_colors[c]/255, label=CLASS_NAMES[c])
               for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=NUM_CLASSES, fontsize=11)
    plt.tight_layout()
    plt.savefig("predictions_v3.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: predictions_v3.png")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "viz":
        # python train_v3.py viz
        visualize_predictions(num_samples=6)
    else:
        # python train_v3.py
        train()