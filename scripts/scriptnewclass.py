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

# Shapefiles
building_path = project_root / "shp-file" / "Built_Up_Area_typ.shp"
road_path = project_root / "shp-file" / "Road.shp"
water_path = project_root / "shp-file" / "Water_Body.shp"

# Load raster
with rasterio.open(tif_path) as src:
    transform = src.transform
    raster_h, raster_w = src.height, src.width
    raster_crs = src.crs
    raster_bounds = src.bounds

# Function to load + clean shapefile
def load_gdf(path):
    gdf = gpd.read_file(path)
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty) & gdf.geometry.is_valid]
    gdf = gdf.cx[raster_bounds.left:raster_bounds.right, raster_bounds.bottom:raster_bounds.top]
    return gdf

# Load all layers
gdf_build = load_gdf(building_path)
gdf_road = load_gdf(road_path)
gdf_water = load_gdf(water_path)

# Ensure at least buildings overlap raster
if gdf_build.empty:
    raise ValueError("No building geometries overlap raster extent.")

# Find centered tile on building extent
minx, miny, maxx, maxy = gdf_build.total_bounds
center_x = (minx + maxx) / 2
center_y = (miny + maxy) / 2
row_center, col_center = rowcol(transform, center_x, center_y)

tile_size = 1024
half = tile_size // 2

row_min = max(0, row_center - half)
col_min = max(0, col_center - half)
row_min = min(row_min, max(0, raster_h - tile_size))
col_min = min(col_min, max(0, raster_w - tile_size))

win_h = min(tile_size, raster_h - row_min)
win_w = min(tile_size, raster_w - col_min)
window = Window(col_off=col_min, row_off=row_min, width=win_w, height=win_h)

# Read image
with rasterio.open(tif_path) as src:
    image = src.read([1, 2, 3], window=window)
    window_transform = src.window_transform(window)

# Create window mask
window_shape = (int(window.height), int(window.width))
mask = np.zeros(window_shape, dtype=np.uint8)

# Rasterize each layer
mask_build = rasterize([(g, 1) for g in gdf_build.geometry],
                       out_shape=window_shape, transform=window_transform, fill=0, dtype=np.uint8)

mask_road = rasterize([(g, 2) for g in gdf_road.geometry],
                      out_shape=window_shape, transform=window_transform, fill=0, dtype=np.uint8)

mask_water = rasterize([(g, 3) for g in gdf_water.geometry],
                       out_shape=window_shape, transform=window_transform, fill=0, dtype=np.uint8)

# Combine masks (priority matters)

# Start with background
mask = np.zeros(window_shape, dtype=np.uint8)

# Priority: water < road < building
mask[mask_water == 3] = 3
mask[mask_road == 2] = 2
mask[mask_build == 1] = 1

mask_crop = mask

# Contrast stretch for display
image_disp = image.astype(np.float32)
p2 = np.percentile(image_disp, 2)
p98 = np.percentile(image_disp, 98)
if p98 > p2:
    image_disp = np.clip((image_disp - p2) / (p98 - p2), 0, 1)
else:
    image_disp = np.zeros_like(image_disp, dtype=np.float32)

# Plot
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(image_disp.transpose(1,2,0))
plt.title("Image")

plt.subplot(1,2,2)
plt.imshow(mask_crop)
plt.title("Multi-Class Mask")

plt.colorbar()  # shows class values

plt.tight_layout()
if 'agg' in plt.get_backend().lower():
    output_path = project_root / "scriptnewclass_preview.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved preview: {output_path}")
else:
    plt.show()