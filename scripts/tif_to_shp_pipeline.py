import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import shape
import torch
import segmentation_models_pytorch as smp


def parse_class_map(class_map_str: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if not class_map_str.strip():
        return mapping

    for part in class_map_str.split(","):
        part = part.strip()
        if not part:
            continue
        class_id_str, class_name = part.split(":", 1)
        mapping[int(class_id_str.strip())] = class_name.strip()
    return mapping


def load_checkpoint(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    mean = np.array(checkpoint.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.array(checkpoint.get("std", [0.25, 0.25, 0.25]), dtype=np.float32)
    class_names = checkpoint.get("class_names", None)

    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned_state[key.replace("model.", "", 1)] = value
        else:
            cleaned_state[key] = value

    return cleaned_state, mean, std, class_names


def build_model(num_classes: int, state_dict: dict, device: str):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def infer_and_vectorize_by_tiles(
    tif_path: Path,
    model,
    mean: np.ndarray,
    std: np.ndarray,
    tile_size: int,
    device: str,
    class_map: dict[int, str],
    min_area: float,
) -> tuple[dict[int, list], object]:
    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width
        crs = src.crs

        geoms_by_class: dict[int, list] = {class_id: [] for class_id in class_map.keys()}

        total_rows = (height + tile_size - 1) // tile_size
        total_cols = (width + tile_size - 1) // tile_size
        total_tiles = total_rows * total_cols
        processed_tiles = 0
        started_at = time.time()

        for row in range(0, height, tile_size):
            for col in range(0, width, tile_size):
                h = min(tile_size, height - row)
                w = min(tile_size, width - col)
                window = Window(col, row, w, h)
                tile_transform = window_transform(window, src.transform)

                tile = src.read([1, 2, 3], window=window).astype(np.float32) / 255.0

                if h != tile_size or w != tile_size:
                    padded = np.zeros((3, tile_size, tile_size), dtype=np.float32)
                    padded[:, :h, :w] = tile
                    tile = padded

                tile_hwc = np.transpose(tile, (1, 2, 0))
                tile_norm = (tile_hwc - mean) / (std + 1e-6)
                tile_chw = np.transpose(tile_norm, (2, 0, 1))

                x = torch.from_numpy(tile_chw).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(x)
                    pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                pred = pred[:h, :w]

                for class_id in class_map.keys():
                    class_mask = (pred == class_id)
                    if not class_mask.any():
                        continue

                    class_mask_uint8 = class_mask.astype(np.uint8)
                    for geom, value in shapes(
                        class_mask_uint8,
                        mask=class_mask,
                        transform=tile_transform,
                    ):
                        if int(value) != 1:
                            continue
                        poly = shape(geom)
                        if poly.is_empty:
                            continue
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                            if poly.is_empty:
                                continue
                        if min_area > 0 and poly.area < min_area:
                            continue
                        geoms_by_class[class_id].append(poly)

                processed_tiles += 1
                if processed_tiles % 100 == 0 or processed_tiles == total_tiles:
                    elapsed = max(time.time() - started_at, 1e-6)
                    tiles_per_sec = processed_tiles / elapsed
                    remaining_tiles = total_tiles - processed_tiles
                    eta_seconds = remaining_tiles / max(tiles_per_sec, 1e-6)
                    progress_pct = (processed_tiles / total_tiles) * 100.0
                    print(
                        f"Progress: {processed_tiles}/{total_tiles} tiles "
                        f"({progress_pct:.1f}%) | ETA: {eta_seconds/60:.1f} min"
                    )

    return geoms_by_class, crs


def geoms_to_shapefile(
    geometries: list,
    class_id: int,
    class_name: str,
    crs,
    output_path: Path,
):
    class_ids = []
    class_names = []

    if geometries:
        class_ids = [class_id] * len(geometries)
        class_names = [class_name] * len(geometries)

    gdf = gpd.GeoDataFrame(
        {
            "class_id": class_ids,
            "class_name": class_names,
            "geometry": geometries,
        },
        crs=crs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="ESRI Shapefile")


def main():
    parser = argparse.ArgumentParser(
        description="Run full-image segmentation on a GeoTIFF and export class-wise shapefiles."
    )
    parser.add_argument("--input-tif", required=True, help="Path to input .tif")
    parser.add_argument("--checkpoint", default="unet_best_v4.pth", help="Model checkpoint path")
    parser.add_argument("--output-dir", default="predicted_shp", help="Output folder for shapefiles")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of classes in model")
    parser.add_argument("--tile-size", type=int, default=256, help="Inference tile size")
    parser.add_argument(
        "--class-map",
        default="1:Built_Up_Area_typ,2:Road,3:Water_Body",
        help="Class mapping in format '1:Built_Up_Area_typ,2:Road,3:Water_Body'",
    )
    parser.add_argument("--min-area", type=float, default=0.0, help="Minimum polygon area to keep")
    parser.add_argument("--device", default="auto", help="cuda | cpu | auto")

    args = parser.parse_args()

    input_tif = Path(args.input_tif)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if not input_tif.exists():
        raise FileNotFoundError(f"Input tif not found: {input_tif}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    class_map = parse_class_map(args.class_map)
    if not class_map:
        raise ValueError("Class map is empty. Provide at least one class mapping.")

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    state_dict, mean, std, ckpt_class_names = load_checkpoint(checkpoint_path, device)
    model = build_model(args.num_classes, state_dict, device)

    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    if ckpt_class_names is not None:
        print(f"Checkpoint class names: {ckpt_class_names}")

    print(f"Running tiled inference on: {input_tif}")
    geoms_by_class, crs = infer_and_vectorize_by_tiles(
        tif_path=input_tif,
        model=model,
        mean=mean,
        std=std,
        tile_size=args.tile_size,
        device=device,
        class_map=class_map,
        min_area=args.min_area,
    )

    print("Exporting shapefiles...")
    for class_id, class_name in class_map.items():
        out_path = output_dir / f"{class_name}.shp"
        geoms_to_shapefile(
            geometries=geoms_by_class.get(class_id, []),
            class_id=class_id,
            class_name=class_name,
            crs=crs,
            output_path=out_path,
        )
        print(f"Saved: {out_path}")

    print("Done. Shapefiles are ready.")


if __name__ == "__main__":
    main()
