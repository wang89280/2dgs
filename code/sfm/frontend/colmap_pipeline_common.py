from __future__ import annotations
from pathlib import Path
import argparse, pycolmap

def make_reader_options(args: argparse.Namespace) -> pycolmap.ImageReaderOptions:
    options = pycolmap.ImageReaderOptions()
    options.camera_model = args.camera_model
    options.default_focal_length_factor = args.focal_factor
    return options

def import_images_to_database(database_path: Path, image_dir: Path, args: argparse.Namespace) -> dict[str, int]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    db = pycolmap.Database.open(str(database_path))
    db.close()
    pycolmap.import_images(
        database_path=str(database_path),
        image_path=str(image_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        options=make_reader_options(args),
    )
    db = pycolmap.Database.open(str(database_path))
    try:
        return {img.name: img.image_id for img in db.read_all_images()}
    finally:
        db.close()
