"""
utils.image_io: Robust image loading and pairing utilities
"""
import csv
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import torch
import torchvision.transforms as T
from PIL import Image


# Supported image extensions
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}


def load_image(path: Union[str, Path], size: Tuple[int, int] = (299, 299)) -> torch.Tensor:
    """
    Load and preprocess an image to tensor format.

    Args:
        path: Path to image file
        size: Target size (height, width) for resizing

    Returns:
        Tensor of shape (C, H, W) with values in [0, 1]
    """
    try:
        with Image.open(path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize and convert to tensor
            transform = T.Compose([
                T.Resize(size),
                T.ToTensor(),  # Converts to [0, 1] range
            ])

            return transform(img)
    except Exception as e:
        logging.error(f"Failed to load image {path}: {e}")
        raise


def find_image_files(directory: Path) -> List[Path]:
    """Find all supported image files in a directory."""
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(image_files)


def match_by_filename(gen_files: List[Path], real_files: List[Path]) -> List[Tuple[Path, Path, str]]:
    """
    Match generated and real images by filename (ignoring extension).

    Returns:
        List of (gen_path, real_path, name) tuples
    """
    real_dict = {f.stem: f for f in real_files}
    pairs = []
    unmatched_gen = []

    for gen_file in gen_files:
        stem = gen_file.stem
        if stem in real_dict:
            pairs.append((gen_file, real_dict[stem], stem))
        else:
            unmatched_gen.append(gen_file)

    if unmatched_gen:
        logging.warning(f"Found {len(unmatched_gen)} unmatched generated images")
        for f in unmatched_gen[:5]:  # Show first 5
            logging.warning(f"  Unmatched: {f.name}")
        if len(unmatched_gen) > 5:
            logging.warning(f"  ... and {len(unmatched_gen) - 5} more")

    unmatched_real = set(real_dict.keys()) - {stem for _, _, stem in pairs}
    if unmatched_real:
        logging.warning(f"Found {len(unmatched_real)} unmatched real images")

    return pairs


def load_pairs_from_csv(manifest_path: Path) -> List[Tuple[Path, Path, str]]:
    """
    Load image pairs from CSV manifest.

    CSV format: gen_path,real_path
    """
    pairs = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            gen_path = Path(row['gen_path'])
            real_path = Path(row['real_path'])
            name = f"pair_{i:04d}"

            if not gen_path.exists():
                logging.warning(f"Generated image not found: {gen_path}")
                continue
            if not real_path.exists():
                logging.warning(f"Real image not found: {real_path}")
                continue

            pairs.append((gen_path, real_path, name))

    return pairs


@dataclass
class LoadedImagePair:
    """Container for paired tensors plus their source paths."""

    gen_tensor: torch.Tensor
    real_tensor: torch.Tensor
    name: str
    gen_path: Path
    real_path: Path


def pair_image_paths(
    gen_dir: Path,
    real_dir: Path,
    strategy: str = "auto",
    manifest: Optional[Path] = None
) -> List[Tuple[Path, Path, str]]:
    """Return matched generated/real image paths without loading pixels."""
    if strategy == "csv":
        if not manifest:
            raise ValueError("Manifest file required for CSV pairing strategy")
        return load_pairs_from_csv(manifest)

    gen_files = find_image_files(gen_dir)
    real_files = find_image_files(real_dir)

    if not gen_files:
        raise ValueError(f"No images found in generated directory: {gen_dir}")
    if not real_files:
        raise ValueError(f"No images found in real directory: {real_dir}")

    return match_by_filename(gen_files, real_files)


def load_and_pair_images_with_paths(
    gen_dir: Path,
    real_dir: Path,
    strategy: str = "auto",
    manifest: Optional[Path] = None,
    image_size: Tuple[int, int] = (299, 299)
) -> List[LoadedImagePair]:
    """Load paired images and keep track of their originating file paths."""
    path_pairs = pair_image_paths(gen_dir, real_dir, strategy=strategy, manifest=manifest)

    if not path_pairs:
        return []

    tensor_pairs: List[LoadedImagePair] = []
    failed_loads = 0

    for gen_path, real_path, name in path_pairs:
        try:
            gen_tensor = load_image(gen_path, image_size)
            real_tensor = load_image(real_path, image_size)
            tensor_pairs.append(
                LoadedImagePair(
                    gen_tensor=gen_tensor,
                    real_tensor=real_tensor,
                    name=name,
                    gen_path=gen_path,
                    real_path=real_path,
                )
            )
        except Exception as e:
            logging.error(f"Failed to load pair {name}: {e}")
            failed_loads += 1

    if failed_loads > 0:
        logging.warning(f"Failed to load {failed_loads} image pairs")

    logging.info(f"Successfully loaded {len(tensor_pairs)} image pairs")
    return tensor_pairs


def load_and_pair_images(
    gen_dir: Path,
    real_dir: Path,
    strategy: str = "auto",
    manifest: Optional[Path] = None,
    warn_unpaired: bool = True,
    image_size: Tuple[int, int] = (299, 299)
) -> List[Tuple[torch.Tensor, torch.Tensor, str]]:
    """
    Backwards-compatible wrapper that returns tensors without path metadata.
    """
    _ = warn_unpaired  # retained for API compatibility
    pairs_with_paths = load_and_pair_images_with_paths(
        gen_dir,
        real_dir,
        strategy=strategy,
        manifest=manifest,
        image_size=image_size,
    )
    return [(p.gen_tensor, p.real_tensor, p.name) for p in pairs_with_paths]
