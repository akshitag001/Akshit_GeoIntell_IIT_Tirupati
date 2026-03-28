import argparse
import subprocess
import sys
from pathlib import Path


def discover_tifs(input_dir: Path) -> list[Path]:
    tif_files = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    return tif_files


def resolve_defaults(model_type: str, checkpoint: str | None) -> tuple[str, str, Path]:
    if model_type == "deeplab":
        default_checkpoint = "deeplab_best_v4.pth"
        default_class_map = "1:Built_Up_Area_typ,2:Road,3:Water_Body"
        pipeline_script = Path("scripts") / "tif_to_shp_pipeline_deeplab.py"
    else:
        default_checkpoint = "unet_best_v4.pth"
        default_class_map = "1:Built_Up_Area_typ,2:Road,3:Water_Body"
        pipeline_script = Path("scripts") / "tif_to_shp_pipeline.py"

    return (checkpoint or default_checkpoint, default_class_map, pipeline_script)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Drop TIFF files in input_tif/ and run this script to generate class-wise shapefiles "
            "for each TIFF automatically."
        )
    )
    parser.add_argument("--input-dir", default="input_tif", help="Folder containing .tif/.tiff files")
    parser.add_argument("--output-root", default="outputs", help="Root folder where per-image shapefile folders are created")
    parser.add_argument("--model-type", choices=["deeplab", "unet"], default="deeplab", help="Which pipeline/model to use")
    parser.add_argument("--checkpoint", default=None, help="Optional custom checkpoint path")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size for inference")
    parser.add_argument("--device", default="auto", help="cuda | cpu | auto")
    parser.add_argument("--include-farm", action="store_true", help="Include class 4: Farm in exported shapefiles")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue processing remaining TIFF files if one fails")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    checkpoint, class_map, pipeline_script = resolve_defaults(args.model_type, args.checkpoint)
    if args.include_farm:
        class_map = "1:Built_Up_Area_typ,2:Road,3:Water_Body,4:Farm"

    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not pipeline_script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")

    tif_files = discover_tifs(input_dir)
    if not tif_files:
        raise FileNotFoundError(
            f"No .tif/.tiff files found in {input_dir}. Put TIFF files there and run again."
        )

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(tif_files)} TIFF file(s) in: {input_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {checkpoint}")
    print("-" * 80)

    failures: list[Path] = []

    for index, tif_path in enumerate(tif_files, start=1):
        image_output_dir = output_root / f"{tif_path.stem}_shp"
        image_output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(pipeline_script),
            "--input-tif",
            str(tif_path),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(image_output_dir),
            "--tile-size",
            str(args.tile_size),
            "--class-map",
            class_map,
            "--device",
            args.device,
        ]

        print(f"[{index}/{len(tif_files)}] Processing: {tif_path.name}")
        print(f"Output folder: {image_output_dir}")

        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failures.append(tif_path)
            print(f"FAILED: {tif_path.name}")
            if not args.continue_on_error:
                break
        else:
            print(f"DONE: {tif_path.name}")

        print("-" * 80)

    if failures:
        print("Completed with failures:")
        for failed in failures:
            print(f"- {failed}")
        sys.exit(1)

    print("All TIFF files processed successfully.")


if __name__ == "__main__":
    main()
