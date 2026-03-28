import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from collections import Counter

# Paths
project_root = Path(__file__).resolve().parents[1]
img_dir = project_root / "dataset/images"
mask_dir = project_root / "dataset/masks"

# -------------------------------
# 🔹 1. Visual Check
# -------------------------------
files = list(img_dir.glob("*.npy"))

sample = random.choice(files)

img = np.load(sample)
mask = np.load(mask_dir / sample.name.replace("img", "mask"))

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img.transpose(1,2,0))
plt.title("Image")

plt.subplot(1,2,2)
plt.imshow(mask)
plt.title("Mask")

plt.show()

# -------------------------------
# 🔹 2. Class Distribution
# -------------------------------
counter = Counter()

for file in list(mask_dir.glob("*.npy"))[:500]:  # limit for speed
    mask = np.load(file)
    counter.update(mask.flatten())

print("Class Distribution:", counter)