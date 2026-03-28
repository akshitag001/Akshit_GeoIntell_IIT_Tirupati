"""
SVAMITVA Drone Image Segmentation — v4 (Complete Rewrite)
==========================================================

Problems fixed from v3 results:
  1. Tiled/patchy artifacts  → Overlap-tile inference + CRF post-processing
  2. Scattered roads         → Road continuity: thin class weight boost + morphological cleanup
  3. Farm ↔ Water confusion  → NDVI/NDWI spectral features added to input
  4. No smooth boundaries    → DeepLabV3+ with ASPP (better global context than UNet)
  5. Class imbalance         → Focal + Dice combined loss (Focal crushes easy background)
  6. No test-time augment    → TTA: flip H/V/both → average predictions

Classes assumed (edit CLASS_NAMES / NUM_CLASSES as needed):
  0 = Background (unlabeled)
  1 = Built-up / Structure  (pink in GT)
  2 = Road                  (green lines in GT)
  3 = Water Body            (large pink blobs in GT)
  4 = Farm / Agriculture    (yellow in your model — currently confused)

Install:
  pip install segmentation-models-pytorch albumentations opencv-python
  pip install pydensecrf   # optional but recommended for CRF post-processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import random
import cv2
import albumentations as A
import segmentation_models_pytorch as smp

# ─────────────────────────────────────────────────────────────
# CONFIG  ← edit these
# ─────────────────────────────────────────────────────────────
IMG_DIR      = "dataset/images"
MASK_DIR     = "dataset/masks"
SAVE_PATH    = "deeplab_best_v4.pth"

NUM_CLASSES  = 5          # 0=bg, 1=builtup, 2=road, 3=water, 4=farm
CLASS_NAMES  = ["Background", "BuiltUp", "Road", "WaterBody", "Farm"]

# If your masks only have 4 classes (no farm separate), set NUM_CLASSES=4
# and update CLASS_NAMES accordingly

TILE_SIZE    = 256        # training patch size
STRIDE       = 128        # overlap stride for inference (half tile = 50% overlap)
BATCH_SIZE   = 8
EPOCHS       = 80
LR           = 2e-4
VAL_SPLIT    = 0.15
PATIENCE     = 20
IGNORE_INDEX = 255        # pixels with label=255 are ignored (unlabeled)

# Class weights multiplier — roads are thin, boost them extra
ROAD_CLASS   = 2          # index of road class
ROAD_BOOST   = 2.5        # multiply road weight by this factor

# ─────────────────────────────────────────────────────────────
# STEP 0: Utility — count pixels & compute stats
# ─────────────────────────────────────────────────────────────
def count_pixels(mask_dir, num_classes=NUM_CLASSES):
    counts = np.zeros(num_classes, dtype=np.int64)
    files  = list(Path(mask_dir).glob("*.npy"))
    print(f"Counting pixels in {len(files)} masks ...")
    for f in files:
        mask = np.load(f)
        for c in range(num_classes):
            counts[c] += int((mask == c).sum())
    print("\n===== PIXEL DISTRIBUTION =====")
    for c in range(num_classes):
        pct = counts[c] / max(counts.sum(), 1) * 100
        print(f"  Class {c:2d} [{CLASS_NAMES[c]:12s}]: {counts[c]:>12,}  ({pct:5.2f}%)")
    print("===============================\n")
    return counts


def compute_dataset_stats(img_dir, num_samples=500):
    files = list(Path(img_dir).glob("*.npy"))
    if len(files) > num_samples:
        files = random.sample(files, num_samples)
    means, stds = [], []
    for f in files:
        img = np.load(f).astype(np.float32)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)          # CHW→HWC
        means.append(img.mean(axis=(0, 1)))
        stds.append(img.std(axis=(0, 1)))
    mean = np.mean(means, axis=0)
    std  = np.mean(stds,  axis=0)
    print(f"Dataset mean : {mean}")
    print(f"Dataset std  : {std}\n")
    return mean, std


# ─────────────────────────────────────────────────────────────
# STEP 1: Spectral indices — helps separate farm vs water
# ─────────────────────────────────────────────────────────────
def add_spectral_indices(img_hwc):
    """
    Add NDVI and NDWI as extra channels.
    Assumes RGB order: channel 0=R, 1=G, 2=B (values 0-255 float).
    Returns (H, W, 5) array: R, G, B, NDVI, NDWI
    """
    R = img_hwc[:, :, 0].astype(np.float32)
    G = img_hwc[:, :, 1].astype(np.float32)
    B = img_hwc[:, :, 2].astype(np.float32)

    # NDVI: (NIR - R) / (NIR + R) — without NIR, approximate with G
    # For true NDVI you'd need a 4th NIR band; this is an RGB proxy
    ndvi = (G - R) / (G + R + 1e-6)          # range ~[-1, 1]
    # NDWI: (G - NIR) / (G + NIR) — proxy using B
    ndwi = (G - B) / (G + B + 1e-6)          # water > 0, land < 0

    ndvi = np.expand_dims(ndvi, -1)
    ndwi = np.expand_dims(ndwi, -1)
    return np.concatenate([img_hwc, ndvi, ndwi], axis=-1)   # (H,W,5)


# ─────────────────────────────────────────────────────────────
# STEP 2: Dataset with spectral features + tile extraction
# ─────────────────────────────────────────────────────────────
class GeoDatasetV4(Dataset):
    def __init__(self, img_dir, mask_dir, mean, std, augment=True,
                 tile_size=TILE_SIZE, use_spectral=True):
        self.img_files    = []
        self.mask_dir     = Path(mask_dir)
        self.mean         = np.array(mean, dtype=np.float32)
        self.std          = np.array(std,  dtype=np.float32)
        self.tile_size    = tile_size
        self.use_spectral = use_spectral

        for img_path in sorted(Path(img_dir).glob("*.npy")):
            idx       = img_path.stem.split("_")[1]
            mask_path = self.mask_dir / f"mask_{idx}.npy"
            if mask_path.exists():
                self.img_files.append(img_path)

        print(f"✅ Paired samples: {len(self.img_files)}")

        if augment:
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.3),
                A.GaussNoise(p=0.2),
                A.ElasticTransform(alpha=50, sigma=5, p=0.2),
                A.GridDistortion(p=0.2),
                A.CoarseDropout(p=0.2),
            ]

            if not self.use_spectral:
                transforms.extend([
                    A.RandomBrightnessContrast(0.25, 0.25, p=0.5),
                    A.HueSaturationValue(10, 20, 10, p=0.3),
                ])

            self.transform = A.Compose(transforms)
        else:
            self.transform = None

    def __len__(self):
        return len(self.img_files)

    def _load(self, idx):
        img_path  = self.img_files[idx]
        idx_num   = img_path.stem.split("_")[1]
        mask_path = self.mask_dir / f"mask_{idx_num}.npy"

        img  = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.int64)

        if img.shape[0] == 3:                     # CHW → HWC
            img = img.transpose(1, 2, 0)

        return img, mask

    def __getitem__(self, idx):
        img, mask = self._load(idx)
        H, W      = img.shape[:2]

        # Random crop to tile_size × tile_size
        if H > self.tile_size and W > self.tile_size:
            y = random.randint(0, H - self.tile_size)
            x = random.randint(0, W - self.tile_size)
            img  = img [y:y+self.tile_size, x:x+self.tile_size]
            mask = mask[y:y+self.tile_size, x:x+self.tile_size]
        else:
            # Pad if smaller
            pad_h = max(0, self.tile_size - H)
            pad_w = max(0, self.tile_size - W)
            img   = np.pad(img,  ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")
            mask  = np.pad(mask, ((0,pad_h),(0,pad_w)),       constant_values=IGNORE_INDEX)
            img   = img [:self.tile_size, :self.tile_size]
            mask  = mask[:self.tile_size, :self.tile_size]

        # Add spectral indices BEFORE normalization (raw RGB values)
        if self.use_spectral:
            img_raw = img.copy()                  # still 0-255 range
            extra   = add_spectral_indices(img_raw)[:, :, 3:]   # (H,W,2)

        # Normalize RGB
        img_norm = (img - self.mean[:3]) / (self.std[:3] + 1e-6)

        if self.use_spectral:
            img_norm = np.concatenate([img_norm, extra], axis=-1)   # (H,W,5)

        # Augment (expects HWC, uint8 mask)
        if self.transform:
            aug  = self.transform(image=img_norm.astype(np.float32),
                                  mask=mask.astype(np.uint8))
            img_norm = aug["image"]
            mask     = aug["mask"].astype(np.int64)

        # HWC → CHW
        img_norm = img_norm.transpose(2, 0, 1)

        return (
            torch.tensor(img_norm, dtype=torch.float32),
            torch.tensor(mask,     dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────
# STEP 3: Model — DeepLabV3+ (better global context than UNet)
# ─────────────────────────────────────────────────────────────
def build_model(num_classes=NUM_CLASSES, in_channels=5):
    """
    DeepLabV3+ with ResNet50 encoder.
    ASPP module captures multi-scale context → better road continuity.
    in_channels=5 because we added NDVI+NDWI to RGB.
    If use_spectral=False, set in_channels=3.
    """
    model = smp.DeepLabV3Plus(
        encoder_name    = "resnet50",
        encoder_weights = "imagenet",
        in_channels     = in_channels,
        classes         = num_classes,
        activation      = None,
    )
    return model


# ─────────────────────────────────────────────────────────────
# STEP 4: Loss Functions
# ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy background pixels,
    forces model to focus on hard/rare classes (roads, water).
    gamma=2 is standard. Higher = more focus on hard examples.
    """
    def __init__(self, weight=None, gamma=2.0, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.gamma        = gamma
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        ce   = F.cross_entropy(pred, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        valid = (target != self.ignore_index)
        return loss[valid].mean()


class SoftDiceLoss(nn.Module):
    """Per-class soft Dice, averaged over foreground classes."""
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX,
                 skip_bg=True):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.skip_bg      = skip_bg

    def forward(self, pred, target):
        pred    = torch.softmax(pred, dim=1)
        tgt_oh  = F.one_hot(target.clamp(0, self.num_classes-1),
                             self.num_classes).permute(0,3,1,2).float()

        # Build valid mask
        if self.ignore_index >= 0:
            valid = (target != self.ignore_index).unsqueeze(1).float()
        else:
            valid = torch.ones_like(pred[:, :1])

        start = 1 if self.skip_bg else 0
        loss  = 0.0
        count = 0
        for c in range(start, self.num_classes):
            p   = pred[:, c] * valid.squeeze(1)
            t   = tgt_oh[:, c] * valid.squeeze(1)
            inter = (p * t).sum(dim=(1,2))
            card  = p.sum(dim=(1,2)) + t.sum(dim=(1,2))
            dice  = (2*inter + 1e-6) / (card + 1e-6)
            loss += (1 - dice.mean())
            count += 1
        return loss / max(count, 1)


class CombinedLoss(nn.Module):
    """30% Focal + 70% Dice — Focal handles imbalance, Dice optimizes IoU"""
    def __init__(self, class_weights, num_classes=NUM_CLASSES):
        super().__init__()
        self.focal = FocalLoss(weight=class_weights)
        self.dice  = SoftDiceLoss(num_classes=num_classes, skip_bg=True)

    def forward(self, pred, target):
        return 0.3 * self.focal(pred, target) + 0.7 * self.dice(pred, target)


# ─────────────────────────────────────────────────────────────
# STEP 5: Class weights (median frequency with road boost)
# ─────────────────────────────────────────────────────────────
def compute_class_weights(pixel_counts, device,
                          road_class=ROAD_CLASS, road_boost=ROAD_BOOST):
    counts = np.array(pixel_counts, dtype=np.float64).copy()
    counts[0] = 0                          # ignore background in weight calc

    freq   = counts / (counts.sum() + 1e-9)
    valid  = freq[freq > 0]
    median = np.median(valid)

    weights = np.where(freq > 0, median / (freq + 1e-9), 0.0)
    weights[0] = 0.1                       # small but nonzero for background

    # Extra boost for road class (thin lines, easy to miss)
    if road_class < len(weights):
        weights[road_class] *= road_boost

    # Normalize to keep loss scale stable
    weights = weights / weights.mean()

    print("Final class weights:")
    for c, w in enumerate(weights):
        print(f"  Class {c} [{CLASS_NAMES[c]:12s}]: {w:.4f}")
    print()

    return torch.tensor(weights, dtype=torch.float32).to(device)


# ─────────────────────────────────────────────────────────────
# STEP 6: Per-class IoU metric
# ─────────────────────────────────────────────────────────────
def per_class_iou(pred_argmax, target, num_classes=NUM_CLASSES):
    """pred_argmax and target are both (H*W,) flattened tensors on CPU."""
    p = pred_argmax.view(-1)
    t = target.view(-1)
    valid = (t != IGNORE_INDEX)
    p, t  = p[valid], t[valid]

    ious = []
    for c in range(num_classes):
        pc = (p == c)
        tc = (t == c)
        inter = (pc & tc).sum().item()
        union = (pc | tc).sum().item()
        ious.append(inter / union if union > 0 else float("nan"))
    return ious


def mean_iou(ious, skip_bg=True):
    start = 1 if skip_bg else 0
    vals  = [v for v in ious[start:] if not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0


# ─────────────────────────────────────────────────────────────
# STEP 7: Test-Time Augmentation (TTA)
# Averages predictions over 4 flips → smoother, more accurate output
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def tta_predict(model, imgs, device):
    """
    4 augmentations: original, flip-H, flip-V, flip-HV
    Returns averaged softmax probabilities.
    """
    model.eval()
    imgs = imgs.to(device)
    preds = torch.softmax(model(imgs), dim=1)

    # Flip H → predict → flip back
    imgs_fh = torch.flip(imgs, dims=[3])
    preds  += torch.flip(torch.softmax(model(imgs_fh), dim=1), dims=[3])

    # Flip V
    imgs_fv = torch.flip(imgs, dims=[2])
    preds  += torch.flip(torch.softmax(model(imgs_fv), dim=1), dims=[2])

    # Flip HV
    imgs_fhv = torch.flip(imgs, dims=[2, 3])
    preds   += torch.flip(torch.softmax(model(imgs_fhv), dim=1), dims=[2, 3])

    return preds / 4.0


# ─────────────────────────────────────────────────────────────
# STEP 8: Post-processing — morphological cleanup
# Smooths roads, removes isolated noise patches
# ─────────────────────────────────────────────────────────────
def morphological_cleanup(pred_mask, road_class=ROAD_CLASS):
    """
    pred_mask: (H,W) numpy array of class indices
    Returns cleaned mask.
    """
    cleaned = pred_mask.copy()

    for cls in range(1, NUM_CLASSES):
        binary = (pred_mask == cls).astype(np.uint8)

        if cls == road_class:
            # Roads: close gaps (connect broken road segments)
            kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            # Remove isolated tiny road blobs < 200px
            binary  = remove_small_blobs(binary, min_size=200)
        else:
            # Other classes: remove tiny isolated patches < 500px
            binary = remove_small_blobs(binary, min_size=500)
            # Smooth boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        cleaned[binary == 1] = cls

    return cleaned


def remove_small_blobs(binary_mask, min_size=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    out = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out[labels == i] = 1
    return out


# ─────────────────────────────────────────────────────────────
# STEP 9: CRF Post-processing (optional — install pydensecrf)
# Sharpens boundaries between classes
# ─────────────────────────────────────────────────────────────
def crf_refine(img_hwc_uint8, pred_probs, num_classes=NUM_CLASSES):
    """
    img_hwc_uint8 : (H,W,3) original RGB image uint8
    pred_probs    : (num_classes, H, W) softmax probabilities float32
    Returns refined (H,W) class mask.
    """
    try:
        import importlib
        dcrf = importlib.import_module("pydensecrf.densecrf")
        unary_from_softmax = importlib.import_module(
            "pydensecrf.utils"
        ).unary_from_softmax
    except ImportError:
        print("pydensecrf not installed — skipping CRF. Run: pip install pydensecrf")
        return np.argmax(pred_probs, axis=0)

    H, W = img_hwc_uint8.shape[:2]
    d    = dcrf.DenseCRF2D(W, H, num_classes)

    unary = unary_from_softmax(pred_probs)
    d.setUnaryEnergy(unary)

    # Appearance kernel: encourages same-color pixels to have same label
    d.addPairwiseBilateral(sxy=10, srgb=13, rgbim=img_hwc_uint8,
                           compat=10, kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    # Spatial smoothness kernel
    d.addPairwiseGaussian(sxy=3, compat=3)

    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape(H, W)


# ─────────────────────────────────────────────────────────────
# STEP 10: Training Loop
# ─────────────────────────────────────────────────────────────
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Classes     : {NUM_CLASSES} → {CLASS_NAMES}")
    print(f"  Model       : DeepLabV3+ / ResNet50")
    print(f"  Loss        : 30% Focal + 70% Dice")
    print(f"  Input ch    : 5 (RGB + NDVI + NDWI)")
    print(f"{'='*60}\n")

    # ── Stats (run once, then hardcode) ──
    mean, std = compute_dataset_stats(IMG_DIR)
    # mean = np.array([X, X, X])
    # std  = np.array([X, X, X])

    pixel_counts = count_pixels(MASK_DIR)
    # pixel_counts = np.array([...])  # hardcode after first run

    # ── Datasets ──
    train_full = GeoDatasetV4(IMG_DIR, MASK_DIR, mean, std, augment=True)
    val_full   = GeoDatasetV4(IMG_DIR, MASK_DIR, mean, std, augment=False)

    n_total = len(train_full)
    n_val   = max(1, int(VAL_SPLIT * n_total))
    n_train = n_total - n_val
    indices = list(range(n_total))
    random.shuffle(indices)

    train_ds = Subset(train_full, indices[:n_train])
    val_ds   = Subset(val_full,  indices[n_train:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}\n")

    # ── Model ──
    model = build_model(num_classes=NUM_CLASSES, in_channels=5).to(device)

    # ── Loss ──
    weights  = compute_class_weights(pixel_counts, device)
    criterion = CombinedLoss(class_weights=weights, num_classes=NUM_CLASSES)

    # ── Optimizer: separate LR for encoder vs decoder ──
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters()
                      if "encoder" not in n]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": LR * 0.1},   # pretrained: smaller LR
        {"params": decoder_params, "lr": LR},          # decoder:   full LR
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6,
        min_lr=1e-7, verbose=True
    )

    scaler     = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    best_miou  = 0.0
    no_improve = 0

    # Header
    header = f"{'Ep':>4} | {'Loss':>7} | {'mIoU':>7} | " + \
             " | ".join(f"{n[:6]:>6}" for n in CLASS_NAMES)
    print(header)
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):

        # ── TRAIN ──
        model.train()
        total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out  = model(imgs)
                loss = criterion(out, masks)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── VALIDATE with TTA ──
        model.eval()
        all_ious = [[] for _ in range(NUM_CLASSES)]

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                probs = tta_predict(model, imgs, device)
                preds = probs.argmax(dim=1).cpu()
                ious  = per_class_iou(preds, masks)
                for c, iou in enumerate(ious):
                    if not np.isnan(iou):
                        all_ious[c].append(iou)

        per_cls = [
            float(np.mean(all_ious[c])) if all_ious[c] else float("nan")
            for c in range(NUM_CLASSES)
        ]
        val_miou = mean_iou(per_cls, skip_bg=True)
        scheduler.step(val_miou)

        lr_now = optimizer.param_groups[1]["lr"]
        pc_str = " | ".join(
            f"{v:6.3f}" if not np.isnan(v) else "   nan"
            for v in per_cls
        )
        print(f"{epoch:>4} | {avg_loss:>7.4f} | {val_miou:>7.4f} | {pc_str}  LR={lr_now:.1e}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "best_miou"  : best_miou,
                "mean"       : mean,
                "std"        : std,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
            }, SAVE_PATH)
            print(f"  ✅ Best model saved — mIoU={best_miou:.4f}")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"\n⏹  Early stopping at epoch {epoch}")
            break

    print(f"\n🏁 Done. Best Val mIoU (no background): {best_miou:.4f}")
    print(f"   Saved to: {SAVE_PATH}")


# ─────────────────────────────────────────────────────────────
# STEP 11: Inference on a full large image (overlap-tile)
# ─────────────────────────────────────────────────────────────
def predict_full_image(img_path, checkpoint_path=SAVE_PATH,
                       tile_size=TILE_SIZE, stride=STRIDE,
                       use_crf=False, use_morph=True):
    """
    Predict on a full-size image using overlapping tiles.
    Overlapping tiles avoid the tile-boundary artifacts you saw.

    img_path: path to .npy file (CHW or HWC, float or uint8)
    """
    ckpt   = torch.load(checkpoint_path, map_location="cpu")
    mean   = ckpt["mean"]
    std    = ckpt["std"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(num_classes=NUM_CLASSES, in_channels=5)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # Load image
    img = np.load(img_path).astype(np.float32)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)                  # CHW → HWC
    H, W = img.shape[:2]
    img_uint8 = img.clip(0, 255).astype(np.uint8)     # keep for CRF

    # Add spectral indices
    img_spec  = add_spectral_indices(img)              # (H,W,5)

    # Normalize RGB channels only
    img_norm       = img_spec.copy()
    img_norm[:,:,:3] = (img_spec[:,:,:3] - mean[:3]) / (std[:3] + 1e-6)

    # Accumulate predictions in overlap-tile fashion
    pred_accum = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map  = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, H - tile_size + 1, stride)) + [H - tile_size]
    xs = list(range(0, W - tile_size + 1, stride)) + [W - tile_size]
    ys = sorted(set(max(0, y) for y in ys))
    xs = sorted(set(max(0, x) for x in xs))

    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = img_norm[y:y+tile_size, x:x+tile_size]   # (T,T,5)
                tile = tile.transpose(2, 0, 1)                   # CHW
                t    = torch.tensor(tile, dtype=torch.float32).unsqueeze(0).to(device)
                prob = torch.softmax(model(t), dim=1).cpu().numpy()[0]  # (C,T,T)
                pred_accum[:, y:y+tile_size, x:x+tile_size] += prob
                count_map   [y:y+tile_size, x:x+tile_size] += 1.0

    # Average overlapping predictions
    pred_accum /= np.maximum(count_map[np.newaxis], 1e-6)
    pred_mask   = np.argmax(pred_accum, axis=0).astype(np.uint8)

    # CRF refinement (sharp boundaries)
    if use_crf:
        pred_mask = crf_refine(img_uint8, pred_accum.astype(np.float32))
        pred_mask = pred_mask.astype(np.uint8)

    # Morphological cleanup (smooth roads, remove noise)
    if use_morph:
        pred_mask = morphological_cleanup(pred_mask)

    return pred_mask


# ─────────────────────────────────────────────────────────────
# STEP 12: Visualization
# ─────────────────────────────────────────────────────────────
def visualize(num_samples=6, checkpoint_path=SAVE_PATH,
              use_morph=True, use_crf=False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    mean = ckpt["mean"]
    std  = ckpt["std"]

    # Color map matching GT colors:
    # 0=black(bg), 1=pink(builtup), 2=green(road), 3=pink-light(water), 4=yellow(farm)
    COLORS = np.array([
        [0,   0,   0  ],     # 0 Background  → black
        [255, 105, 180],     # 1 Built-up    → pink
        [0,   200, 0  ],     # 2 Road        → green
        [255, 182, 193],     # 3 Water Body  → light pink
        [255, 215, 0  ],     # 4 Farm        → yellow
    ], dtype=np.uint8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(num_classes=NUM_CLASSES, in_channels=5).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds      = GeoDatasetV4(IMG_DIR, MASK_DIR, mean, std, augment=False)
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))

    fig, axes = plt.subplots(num_samples, 3, figsize=(14, 5*num_samples))
    if num_samples == 1: axes = [axes]

    for row, idx in enumerate(indices):
        img_t, mask = ds[idx]
        with torch.no_grad():
            probs = tta_predict(model, img_t.unsqueeze(0), device)
            pred  = probs.argmax(dim=1).squeeze(0).cpu().numpy()

        if use_morph:
            pred = morphological_cleanup(pred)

        # Denorm for display
        img_np = img_t[:3].numpy().transpose(1,2,0)
        img_np = (img_np * std[:3] + mean[:3]).clip(0, 255).astype(np.uint8)

        gt_rgb   = COLORS[mask.numpy().clip(0, NUM_CLASSES-1)]
        pred_rgb = COLORS[pred.clip(0, NUM_CLASSES-1)]

        axes[row][0].imshow(img_np);   axes[row][0].set_title("Satellite", fontsize=11)
        axes[row][1].imshow(gt_rgb);   axes[row][1].set_title("Ground Truth", fontsize=11)
        axes[row][2].imshow(pred_rgb); axes[row][2].set_title("Prediction (v4)", fontsize=11)
        for ax in axes[row]: ax.axis("off")

    patches = [mpatches.Patch(color=COLORS[c]/255, label=CLASS_NAMES[c])
               for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=NUM_CLASSES, fontsize=11)
    plt.tight_layout()
    plt.savefig("predictions_v4.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: predictions_v4.png")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "viz":
        visualize(num_samples=6)
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        # python train_v4.py predict path/to/image.npy
        img_path = sys.argv[2]
        mask     = predict_full_image(img_path, use_morph=True, use_crf=False)
        np.save("predicted_mask.npy", mask)
        print(f"Saved predicted_mask.npy  shape={mask.shape}")
    else:
        train()