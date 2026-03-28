import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import geopandas as gpd
import numpy as np
from pathlib import Path
import os

# ROOT
project_root = Path(__file__).resolve().parents[1]

# INPUT FOLDERS
tif_folder = project_root
shp_folder = project_root / "shp-file"

# SHP FILES
building_path = shp_folder / "Built_Up_Area_typ.shp"
road_path = shp_folder / "Road.shp"
water_path = shp_folder / "Water_Body.shp"

# OUTPUT
img_dir = project_root / "dataset/images"
mask_dir = project_root / "dataset/masks"

os.makedirs(img_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# LOAD SHAPEFILES FUNCTION
def load_gdf(path, raster_crs, raster_bounds):
    gdf = gpd.read_file(path)

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty) & gdf.geometry.is_valid]

    # 🔥 IMPORTANT: filter per raster
    gdf = gdf.cx[
        raster_bounds.left:raster_bounds.right,
        raster_bounds.bottom:raster_bounds.top
    ]

    return gdf

tile_size = 256
count = 0

# 🔥 LOOP THROUGH ALL TIF FILES
for tif_path in tif_folder.glob("*.tif"):

    print(f"\nProcessing: {tif_path.name}")

    with rasterio.open(tif_path) as src:

        raster_crs = src.crs
        bounds = src.bounds

        # Load shapefiles aligned to this tif
        gdf_build = load_gdf(building_path, raster_crs, bounds)
        gdf_road  = load_gdf(road_path, raster_crs, bounds)
        gdf_water = load_gdf(water_path, raster_crs, bounds)

        print(f"Buildings: {len(gdf_build)}, Roads: {len(gdf_road)}, Water: {len(gdf_water)}")

        # 🔥 TILE LOOP (NO FULL IMAGE LOAD)
        for i in range(0, src.height, tile_size):
            for j in range(0, src.width, tile_size):

                # skip edge tiles
                if i + tile_size > src.height or j + tile_size > src.width:
                    continue

                window = Window(j, i, tile_size, tile_size)

                # read tile
                img_tile = src.read([1, 2, 3], window=window)
                window_transform = src.window_transform(window)

                # 🔥 rasterize per tile (CRITICAL FIX)
                mask_build = rasterize(
                    [(g, 1) for g in gdf_build.geometry],
                    out_shape=(tile_size, tile_size),
                    transform=window_transform,
                    fill=0,
                    dtype=np.uint8
                )

                mask_road = rasterize(
                    [(g, 2) for g in gdf_road.geometry],
                    out_shape=(tile_size, tile_size),
                    transform=window_transform,
                    fill=0,
                    dtype=np.uint8
                )

                mask_water = rasterize(
                    [(g, 3) for g in gdf_water.geometry],
                    out_shape=(tile_size, tile_size),
                    transform=window_transform,
                    fill=0,
                    dtype=np.uint8
                )

                # merge masks (priority)
                mask_tile = np.zeros((tile_size, tile_size), dtype=np.uint8)
                mask_tile[mask_water == 3] = 3
                mask_tile[mask_road == 2] = 2
                mask_tile[mask_build == 1] = 1

                # skip empty tiles
                if np.sum(mask_tile) == 0:
                    continue

                # normalize
                img_tile = img_tile / 255.0

                # save
                np.save(img_dir / f"img_{count}.npy", img_tile)
                np.save(mask_dir / f"mask_{count}.npy", mask_tile)

                count += 1

    print(f"Tiles generated so far: {count}")

print(f"\n🔥 DONE! Total tiles: {count}")