import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.transform import rowcol
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
project_root = Path(__file__).resolve().parents[1]
tif_path = project_root / "TIMMOWAL_37695_ORI.tif"
shp_path = project_root / "shp-file" / "Built_Up_Area_typ.shp"

# Load raster
with rasterio.open(tif_path) as src:
    transform = src.transform
    raster_crs = src.crs
    raster_bounds = src.bounds

# Load shapefile
gdf = gpd.read_file(shp_path)

# Fix CRS
if gdf.crs != raster_crs:
    gdf = gdf.to_crs(raster_crs)

# Remove invalid geometries
gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty) & gdf.geometry.is_valid]

# Keep only geometries that overlap raster extent
gdf = gdf.cx[raster_bounds.left:raster_bounds.right, raster_bounds.bottom:raster_bounds.top]
if gdf.empty:
    raise ValueError("No building geometries overlap raster extent.")

# Get bounds of overlapping buildings
minx, miny, maxx, maxy = gdf.total_bounds
print("Bounds:", minx, miny, maxx, maxy)

# Build fixed-size tile around building-center to avoid huge allocations
with rasterio.open(tif_path) as src:
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    center_row, center_col = rowcol(src.transform, center_x, center_y)

    tile_size = 1024
    half = tile_size // 2
    row_min = max(0, center_row - half)
    col_min = max(0, center_col - half)
    row_min = min(row_min, max(0, src.height - tile_size))
    col_min = min(col_min, max(0, src.width - tile_size))

    win_h = min(tile_size, src.height - row_min)
    win_w = min(tile_size, src.width - col_min)
    window = Window(col_off=col_min, row_off=row_min, width=win_w, height=win_h)

    image = src.read([1, 2, 3], window=window)
    window_transform = src.window_transform(window)

out_shape = (int(window.height), int(window.width))
print("Window size:", out_shape)

# Rasterize directly in window space
mask_crop = rasterize(
    [(geom, 1) for geom in gdf.geometry],
    out_shape=out_shape,
    transform=window_transform,
    fill=0,
    dtype=np.uint8
)
print("Mask pixels:", int(mask_crop.sum()))

# Plot
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(image.transpose(1,2,0))
plt.title("Image (Building Area)")

plt.subplot(1,2,2)
plt.imshow(mask_crop, cmap='gray')
plt.title("Mask (Buildings)")

plt.tight_layout()
if 'agg' in plt.get_backend().lower():
    output_path = project_root / "script3_preview.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved preview: {output_path}")
else:
    plt.show()