#!/usr/bin/env python3
"""
Semantic Consistency Evaluation for Image-to-Image Translation
Uses DeepLabV3 (ResNet backbone) to compare segmentation masks between
source and translated images.

Author: Research Script
Date: November 2025
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import warnings

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# Cityscapes color palette (19 classes + background) for visualization
CITYSCAPES_COLORS = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32], [  0,   0,   0]
], dtype=np.uint8)

CITYSCAPES_CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

def colorize_mask(mask: np.ndarray, palette: np.ndarray = CITYSCAPES_COLORS) -> Image.Image:
    """Convert a segmentation mask to a colorized PIL image using the palette."""
    if mask.ndim != 2:
        raise ValueError("Segmentation mask must be 2D for colorization")
    palette_len = palette.shape[0]
    clipped = np.clip(mask, 0, palette_len - 1)
    colored = palette[clipped]
    return Image.fromarray(colored, mode='RGB')


def build_output_path(
    root: Optional[Path],
    base_dir: Path,
    image_path: Path,
    suffix: str
) -> Optional[Path]:
    """Construct an output path mirroring the image path under the provided root."""
    if root is None:
        return None
    relative_path = image_path.relative_to(base_dir)
    return (root / relative_path).with_suffix(suffix)


def load_suffixes(config_path: Optional[Path]) -> Tuple[str, ...]:
    """Load default suffixes from a JSON config file."""
    if config_path is None:
        return ()
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Suffix config not found at {config_path}. Using empty defaults.")
        return ()
    except json.JSONDecodeError as err:
        print(f"Warning: Failed to parse suffix config {config_path}: {err}. Using empty defaults.")
        return ()

    suffixes = data.get('suffixes', [])
    if not isinstance(suffixes, list):
        print(f"Warning: 'suffixes' must be a list in {config_path}. Using empty defaults.")
        return ()

    normalized = []
    for entry in suffixes:
        if isinstance(entry, str):
            cleaned = entry.strip()
            if cleaned:
                normalized.append(cleaned)
    return tuple(normalized)

class SegFormerEvaluator:
    """
    Evaluator for semantic consistency using SegFormer.
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize SegFormer model and processor.
        
        Args:
            model_name: HuggingFace model identifier
            device: Computation device ('cuda' or 'cpu'). Auto-detect if None.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names if class_names else CITYSCAPES_CLASS_NAMES
        
        print(f"Initializing SegFormer ({model_name}) on {self.device}...")
        
        # Load processor and model
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get number of classes from config
        self.num_classes = self.model.config.num_labels
        
        print(f"Model loaded with {self.num_classes} classes!\n")
    
    def segment_image(self, image_path: Path) -> np.ndarray:
        """
        Perform semantic segmentation on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Segmentation mask as numpy array (H, W) with class indices
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Upsample to original size and get predictions
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=(original_size[1], original_size[0]),  # (H, W)
                mode='bilinear',
                align_corners=False
            )
            
            predictions = upsampled_logits.argmax(dim=1).squeeze(0)
        
        return predictions.cpu().numpy()
    
    @staticmethod
    def compute_pixel_accuracy(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute pixel-wise accuracy between two segmentation masks."""
        assert mask1.shape == mask2.shape, "Masks must have the same shape"
        correct_pixels = np.sum(mask1 == mask2)
        total_pixels = mask1.size
        return (correct_pixels / total_pixels) * 100.0
    
    @staticmethod
    def compute_iou(
        mask1: np.ndarray,
        mask2: np.ndarray,
        num_classes: int,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute IoU metrics."""
        assert mask1.shape == mask2.shape, "Masks must have the same shape"
        
        ious = []
        class_ious = {}
        
        for cls in range(num_classes):
            mask1_cls = (mask1 == cls)
            mask2_cls = (mask2 == cls)
            
            intersection = np.logical_and(mask1_cls, mask2_cls).sum()
            union = np.logical_or(mask1_cls, mask2_cls).sum()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
                label = (
                    class_names[cls]
                    if class_names and cls < len(class_names)
                    else f'class_{cls}'
                )
                class_ious[label] = iou
        
        miou = np.mean(ious) if ious else 0.0
        
        return {
            'mIoU': miou,
            'class_IoUs': class_ious
        }
    
    def evaluate_pair(
        self,
        source_path: Path,
        translated_path: Path
    ) -> Dict[str, float]:
        """Evaluate semantic consistency for a single image pair."""
        # Segment both images
        source_mask = self.segment_image(source_path)
        translated_mask = self.segment_image(translated_path)
        
        # Compute metrics
        pixel_acc = self.compute_pixel_accuracy(source_mask, translated_mask)
        iou_metrics = self.compute_iou(
            source_mask,
            translated_mask,
            self.num_classes,
            self.class_names
        )
        
        return {
            'pixel_accuracy': pixel_acc,
            'mIoU': iou_metrics['mIoU'] * 100.0,
            'class_IoUs': iou_metrics['class_IoUs']
        }

class DeepLabV3Evaluator:
    """
    Evaluator for semantic consistency using DeepLabV3 segmentation.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet101',
        device: Optional[str] = None,
        num_classes: int = 19,  # Cityscapes has 19 classes + background
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the DeepLabV3 model and preprocessing pipeline.
        
        Args:
            backbone: Model backbone ('resnet50' or 'resnet101')
            device: Computation device ('cuda' or 'cpu'). Auto-detect if None.
            num_classes: Number of segmentation classes (20 for Cityscapes)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.class_names = class_names if class_names else CITYSCAPES_CLASS_NAMES
        
        print(f"Initializing DeepLabV3 with {backbone} backbone on {self.device}...")
        
        # Load pre-trained DeepLabV3 model
        if backbone == 'resnet50':
            self.model = models.segmentation.deeplabv3_resnet50(
                pretrained=False,
                progress=True, 
                num_classes=self.num_classes                
            )
            state_dict = torch.load(
                'weights/deeplabv3_resnet50_cityscapes.bin',
                map_location=self.device
            )
            # Some checkpoints include auxiliary heads; load non-matching keys loosely.
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if unexpected:
                print(f"Warning: Ignored unexpected keys in checkpoint: {unexpected}")
            if missing:
                print(f"Warning: Missing keys when loading checkpoint: {missing}")
            self.model.eval()
        elif backbone == 'resnet101':
            self.model = models.segmentation.deeplabv3_resnet101(
                pretrained=True,
                progress=True, 
                num_classes=self.num_classes
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet50' or 'resnet101'.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing transforms (ImageNet normalization)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("Model loaded successfully!\n")
    
    def segment_image(self, image_path: Path) -> np.ndarray:
        """
        Perform semantic segmentation on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Segmentation mask as numpy array (H, W) with class indices
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)['out']
            
            # Get class predictions (argmax over channels)
            predictions = torch.argmax(output, dim=1).squeeze(0)
            
            # Resize to original image size if needed
            if predictions.shape != (original_size[1], original_size[0]):
                predictions = F.interpolate(
                    predictions.unsqueeze(0).unsqueeze(0).float(),
                    size=(original_size[1], original_size[0]),
                    mode='nearest'
                ).squeeze().long()
        
        return predictions.cpu().numpy()
    
    @staticmethod
    def compute_pixel_accuracy(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute pixel-wise accuracy between two segmentation masks.
        
        Args:
            mask1: First segmentation mask (H, W)
            mask2: Second segmentation mask (H, W)
            
        Returns:
            Pixel accuracy as percentage (0-100)
        """
        assert mask1.shape == mask2.shape, "Masks must have the same shape"
        correct_pixels = np.sum(mask1 == mask2)
        total_pixels = mask1.size
        return (correct_pixels / total_pixels) * 100.0
    
    @staticmethod
    def compute_iou(
        mask1: np.ndarray,
        mask2: np.ndarray,
        num_classes: int,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute Intersection-over-Union (IoU) metrics.
        
        Args:
            mask1: First segmentation mask (H, W)
            mask2: Second segmentation mask (H, W)
            num_classes: Number of segmentation classes
            
        Returns:
            Dictionary with mIoU and class-wise IoU values
        """
        assert mask1.shape == mask2.shape, "Masks must have the same shape"
        
        ious = []
        class_ious = {}
        
        for cls in range(num_classes):
            # Get binary masks for current class
            mask1_cls = (mask1 == cls)
            mask2_cls = (mask2 == cls)
            
            # Compute intersection and union
            intersection = np.logical_and(mask1_cls, mask2_cls).sum()
            union = np.logical_or(mask1_cls, mask2_cls).sum()
            
            # Compute IoU (skip if class not present in either mask)
            if union > 0:
                iou = intersection / union
                ious.append(iou)
                label = (
                    class_names[cls]
                    if class_names and cls < len(class_names)
                    else f'class_{cls}'
                )
                class_ious[label] = iou
        
        # Compute mean IoU
        miou = np.mean(ious) if ious else 0.0
        
        return {
            'mIoU': miou,
            'class_IoUs': class_ious
        }
    
    @staticmethod
    def compute_frequency_weighted_iou(
        mask1: np.ndarray,
        mask2: np.ndarray,
        num_classes: int,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute frequency-weighted IoU, accounting for class imbalance.
        
        Args:
            mask1: First segmentation mask (H, W)
            mask2: Second segmentation mask (H, W)
            num_classes: Number of semantic classes
            class_names: Optional list of class labels for reporting
            
        Returns:
            Dictionary with fw-IoU, mIoU, and class frequencies (values may be nested)
        """

        assert mask1.shape == mask2.shape, "Masks must have the same shape"
        
        class_ious = []
        class_frequencies = []
        class_iou_dict = {}
        
        total_pixels = mask1.size
        
        for cls in range(num_classes):
            # Get binary masks for current class
            mask1_cls = (mask1 == cls)
            mask2_cls = (mask2 == cls)
            
            # Compute intersection and union
            intersection = np.logical_and(mask1_cls, mask2_cls).sum()
            union = np.logical_or(mask1_cls, mask2_cls).sum()
            
            # Frequency of class in ground truth (mask1)
            frequency = mask1_cls.sum() / total_pixels
            
            # Compute IoU (skip if class not present in either mask)
            if union > 0:
                iou = intersection / union
                class_ious.append(iou)
                class_frequencies.append(frequency)
                label = (
                    class_names[cls]
                    if class_names and cls < len(class_names)
                    else f'class_{cls}'
                )
                class_iou_dict[label] = {
                    'IoU': float(iou),
                    'frequency': float(frequency)
                }
        
        # Compute metrics
        miou = float(np.mean(class_ious)) if class_ious else 0.0
        
        # Frequency-weighted IoU
        if class_frequencies:
            fw_iou = float(
                np.sum(np.array(class_frequencies) * np.array(class_ious)) / np.sum(class_frequencies)
            )
        else:
            fw_iou = 0.0
        
        return {
            'mIoU': miou * 100.0,
            'fw_IoU': fw_iou * 100.0,
            'class_details': class_iou_dict
        }

    def evaluate_pair(
        self,
        source_path: Path,
        translated_path: Path
    ) -> Dict[str, Any]:
        """
        Evaluate semantic consistency for a single image pair.
        
        Args:
            source_path: Path to source image
            translated_path: Path to translated image
            
        Returns:
            Dictionary with consistency metrics
        """
        # Segment both images
        source_mask = self.segment_image(source_path)
        translated_mask = self.segment_image(translated_path)
        
        # Compute metrics
        pixel_acc = self.compute_pixel_accuracy(source_mask, translated_mask)
        iou_metrics = self.compute_iou(
            source_mask,
            translated_mask,
            self.num_classes,
            self.class_names
        )
        
        return {
            'pixel_accuracy': pixel_acc,
            'mIoU': iou_metrics['mIoU'] * 100.0,  # Convert to percentage
            'class_IoUs': iou_metrics['class_IoUs']
        }


def get_image_pairs(
    source_dir: Path,
    translated_dir: Path,
    extensions: Tuple[str, ...] = ('.png',),
    strip_suffixes: Tuple[str, ...] = (),
    default_suffixes: Tuple[str, ...] = ()
) -> List[Tuple[Path, Path]]:
    """
    Find matching image pairs by mirroring directory structures and
    tolerating filename suffixes (e.g., *_ref_anon).
    
    Args:
        source_dir: Directory with source images
        translated_dir: Directory with translated images
        extensions: Allowed image file extensions
        strip_suffixes: Filename suffixes to drop before matching
        
    Returns:
        List of (source_path, translated_path) tuples
    """
    extensions = tuple(sorted({ext.lower() for ext in extensions})) if extensions else ('.png',)
    combined_suffixes = (strip_suffixes or ()) + default_suffixes
    suffix_order = [suffix for suffix in dict.fromkeys(combined_suffixes) if suffix]
    suffixes: Tuple[str, ...] = tuple(sorted(suffix_order, key=len, reverse=True))
    separators = ('_', '-', '.', ' ')

    def normalize_stem(stem: str) -> str:
        """Strip configured suffixes (and adjoining separators) from a filename stem."""
        base = stem
        changed = True
        while changed:
            changed = False
            for suffix in suffixes:
                if not suffix or not base.endswith(suffix):
                    continue
                trimmed = base[:-len(suffix)]
                while trimmed and trimmed[-1] in separators:
                    trimmed = trimmed[:-1]
                base = trimmed
                changed = True
                break
        return base

    def iter_images(root: Path) -> List[Path]:
        if not root.exists():
            return []
        return sorted(
            path for path in root.rglob('*')
            if path.is_file() and path.suffix.lower() in extensions
        )

    def dir_key(base: Path, path: Path) -> str:
        rel_parent = path.relative_to(base).parent
        return rel_parent.as_posix() if rel_parent != Path('.') else '.'

    def build_entries(root: Path) -> List[Tuple[Path, str, str]]:
        entries: List[Tuple[Path, str, str]] = []
        for img_path in iter_images(root):
            entries.append((
                img_path,
                dir_key(root, img_path),
                normalize_stem(img_path.stem)
            ))
        return entries

    source_entries = build_entries(source_dir)
    translated_entries = build_entries(translated_dir)

    translated_by_dir: Dict[Tuple[str, str], List[Path]] = {}
    translated_by_stem: Dict[str, List[Path]] = {}
    for path, rel_dir, norm_stem in translated_entries:
        translated_by_dir.setdefault((rel_dir, norm_stem), []).append(path)
        translated_by_stem.setdefault(norm_stem, []).append(path)

    print(f"Found {len(source_entries)} source images for pairing.")

    pairs: List[Tuple[Path, Path]] = []
    missing = 0
    fallback_matches = 0

    for source_path, rel_dir, norm_stem in source_entries:
        key = (rel_dir, norm_stem)
        candidates = translated_by_dir.get(key)
        used_fallback = False

        if not candidates:
            candidates = translated_by_stem.get(norm_stem, [])
            used_fallback = bool(candidates)

        if not candidates:
            missing += 1
            print(f"Warning: No matching translated image for {source_path.relative_to(source_dir)}")
            continue

        match = next(
            (candidate for candidate in candidates
             if candidate.suffix.lower() == source_path.suffix.lower()),
            candidates[0]
        )

        if used_fallback:
            fallback_matches += 1
            print(
                f"Info: Matched {source_path.relative_to(source_dir)} using fallback search in translated tree."
            )

        pairs.append((source_path, match))

    print(
        f"Pairing complete: {len(pairs)} matches, {missing} missing, {fallback_matches} fallback matches."
    )

    return pairs


def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across all image pairs.
    
    Args:
        all_results: List of result dictionaries from each pair
        
    Returns:
        Dictionary with averaged metrics
    """
    if not all_results:
        return {}
    
    # Aggregate pixel accuracy and mIoU
    avg_pixel_acc = np.mean([r['pixel_accuracy'] for r in all_results])
    avg_miou = np.mean([r['mIoU'] for r in all_results])
    
    # Aggregate class-wise IoU (average across pairs where class appears)
    all_class_ious = {}
    for result in all_results:
        for cls, iou in result['class_IoUs'].items():
            if cls not in all_class_ious:
                all_class_ious[cls] = []
            all_class_ious[cls].append(iou)
    
    avg_class_ious = {
        cls: np.mean(ious) * 100.0  # Convert to percentage
        for cls, ious in all_class_ious.items()
    }
    
    return {
        'average_pixel_accuracy': avg_pixel_acc,
        'average_mIoU': avg_miou,
        'average_class_IoUs': avg_class_ious,
        'num_pairs_evaluated': len(all_results)
    }


def save_results(
    results: Dict[str, Any],
    output_path: Path,
    detailed_results: Optional[List[Dict[str, Any]]] = None,
    source_dir: Optional[Path] = None,
    translated_dir: Optional[Path] = None
):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Aggregated results dictionary
        output_path: Path to output JSON file
        detailed_results: Optional list of per-pair results
    """
    output_data: Dict[str, Any] = {
        'summary': results,
        'meta': {
            'source_dir': str(source_dir) if source_dir else None,
            'translated_dir': str(translated_dir) if translated_dir else None,
            'generated_at': datetime.now().isoformat()
        }
    }
    
    if detailed_results:
        output_data['detailed_results'] = detailed_results
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_results(
    results: Dict[str, Any],
    source_dir: Optional[Path] = None,
    translated_dir: Optional[Path] = None
):
    """
    Print formatted evaluation results.
    
    Args:
        results: Aggregated results dictionary
    """
    print("\n" + "="*60)
    print("SEMANTIC CONSISTENCY EVALUATION RESULTS")
    print("="*60)
    if source_dir:
        print(f"Source directory: {source_dir}")
    if translated_dir:
        print(f"Translated directory: {translated_dir}")
    print(f"\nNumber of image pairs evaluated: {results['num_pairs_evaluated']}")
    print(f"\nAverage Pixel Accuracy: {results['average_pixel_accuracy']:.2f}%")
    print(f"Average mIoU: {results['average_mIoU']:.2f}%")
    
    if results.get('average_class_IoUs'):
        print("\nClass-wise IoU (%):")
        for cls, iou in sorted(results['average_class_IoUs'].items()):
            print(f"  {cls}: {iou:.2f}%")
    
    print("="*60 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Evaluate semantic consistency in image-to-image translation using SegFormer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        help='Directory containing source images'
    )
    parser.add_argument(
        '--translated_dir',
        type=str,
        required=True,
        help='Directory containing translated images (matching filenames)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--backbone',
        type=str,
        choices=['resnet50', 'resnet101'],
        default='resnet50',
        help='DeepLabV3 backbone architecture (default: resnet101)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save evaluation results (default: ./results)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Computation device (default: auto-detect)'
    )
    parser.add_argument(
        '--save_detailed',
        action='store_true',
        help='Save detailed per-pair results in output JSON'
    )
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.png', '.jpg', '.jpeg', '.bmp'],
        help='Image file extensions to process (default: .png .jpg .jpeg .bmp)'
    )
    parser.add_argument(
        '--strip_suffixes',
        type=str,
        nargs='+',
        default=None,
        help='Filename suffixes to strip from both source and translated stems before matching'
    )
    parser.add_argument(
        '--suffix_config',
        type=str,
        default='configs/suffixes.json',
        help='Path to JSON file listing default suffixes to strip when pairing filenames'
    )
    parser.add_argument(
        '--source_cache_dir',
        type=str,
        default='cache/',
        help='Directory to cache source segmentation masks (*.npy) for reuse'
    )
    parser.add_argument(
        '--reuse_cached_source',
        action='store_true',
        help='Load cached source masks from --source_cache_dir when available'
    )
    parser.add_argument(
        '--save_segmentations_dir',
        type=str,
        default=None,
        help='Directory to save raw segmentation masks (*.npy) for source and translated images'
    )
    parser.add_argument(
        '--save_color_segmentations_dir',
        type=str,
        default=None,
        help='Directory to save colorized segmentation previews (*.png)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=[
            'segformer-b0',
            'segformer-b1', 
            'segformer-b2',
            'segformer-b3',
            'segformer-b4',
            'segformer-b5'
        ],
        default='segformer-b5',
        help='SegFormer model size (default: b5 for best accuracy)'
    )
    
    args = parser.parse_args()
    
    # Map model choice to HuggingFace model name
    model_mapping = {
        'segformer-b0': 'nvidia/segformer-b0-finetuned-cityscapes-768-768',
        'segformer-b1': 'nvidia/segformer-b1-finetuned-cityscapes-1024-1024',
        'segformer-b2': 'nvidia/segformer-b2-finetuned-cityscapes-1024-1024',
        'segformer-b3': 'nvidia/segformer-b3-finetuned-cityscapes-1024-1024',
        'segformer-b4': 'nvidia/segformer-b4-finetuned-cityscapes-1024-1024',
        'segformer-b5': 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024',
    }
    
    
    # Convert paths
    source_dir = Path(args.source_dir)
    translated_dir = Path(args.translated_dir)
    output_dir = Path(args.output_dir)
    suffix_config_path = Path(args.suffix_config) if args.suffix_config else None
    source_cache_dir = Path(args.source_cache_dir) if args.source_cache_dir else None
    seg_save_dir = Path(args.save_segmentations_dir) if args.save_segmentations_dir else None
    color_save_dir = Path(args.save_color_segmentations_dir) if args.save_color_segmentations_dir else None
    seg_source_root = (seg_save_dir / 'source') if seg_save_dir else None
    seg_translated_root = (seg_save_dir / 'translated') if seg_save_dir else None
    color_source_root = (color_save_dir / 'source') if color_save_dir else None
    color_translated_root = (color_save_dir / 'translated') if color_save_dir else None
    
    # Validate directories
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not translated_dir.exists():
        raise FileNotFoundError(f"Translated directory not found: {translated_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    for optional_dir in (
        source_cache_dir,
        seg_save_dir,
        seg_source_root,
        seg_translated_root,
        color_save_dir,
        color_source_root,
        color_translated_root,
    ):
        if optional_dir:
            optional_dir.mkdir(parents=True, exist_ok=True)

    def save_mask_array(mask: np.ndarray, root: Optional[Path], base_dir: Path, image_path: Path):
        if root is None:
            return
        target_path = build_output_path(root, base_dir, image_path, '.npy')
        if target_path is None:
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, mask.astype(np.uint8))

    def save_color_preview(mask: np.ndarray, root: Optional[Path], base_dir: Path, image_path: Path):
        if root is None:
            return
        target_path = build_output_path(root, base_dir, image_path, '.png')
        if target_path is None:
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        colorize_mask(mask).save(target_path)
    
    # Initialize evaluator

    model_name = model_mapping[args.model]    
    evaluator = SegFormerEvaluator(
        model_name=model_name,
        device=args.device
    )
    
    # Get image pairs
    print("Finding image pairs...")
    default_suffixes = load_suffixes(suffix_config_path)

    image_pairs = get_image_pairs(
        source_dir,
        translated_dir,
        extensions=tuple(args.extensions),
        strip_suffixes=tuple(args.strip_suffixes) if args.strip_suffixes else (),
        default_suffixes=default_suffixes
    )
    
    if not image_pairs:
        print("Error: No matching image pairs found!")
        return
    
    print(f"Found {len(image_pairs)} image pairs\n")
    
    # Evaluate all pairs
    all_results = []
    cache_hits = 0
    cache_writes = 0
    
    print("Evaluating semantic consistency...")
    for source_path, translated_path in tqdm(image_pairs, desc="Processing"):
        try:
            cache_file = build_output_path(source_cache_dir, source_dir, source_path, '.npy')
            source_mask: np.ndarray
            if cache_file and args.reuse_cached_source and cache_file.exists():
                source_mask = np.load(cache_file, allow_pickle=False)
                cache_hits += 1
            else:
                source_mask = evaluator.segment_image(source_path)
                if cache_file:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache_file, source_mask.astype(np.uint8))
                    cache_writes += 1
            translated_mask = evaluator.segment_image(translated_path)

            save_mask_array(source_mask, seg_source_root, source_dir, source_path)
            save_mask_array(translated_mask, seg_translated_root, translated_dir, translated_path)
            save_color_preview(source_mask, color_source_root, source_dir, source_path)
            save_color_preview(translated_mask, color_translated_root, translated_dir, translated_path)

            pixel_acc = evaluator.compute_pixel_accuracy(source_mask, translated_mask)
            iou_metrics = evaluator.compute_iou(source_mask, translated_mask, evaluator.num_classes)
            result = {
                'pixel_accuracy': pixel_acc,
                'mIoU': iou_metrics['mIoU'] * 100.0,
                'class_IoUs': iou_metrics['class_IoUs'],
                'source_image': source_path.name,
                'translated_image': translated_path.name
            }
            all_results.append(result)
        except Exception as e:
            print(f"\nError processing {source_path.name}: {str(e)}")
            continue

    if source_cache_dir:
        print(
            f"Source cache summary -> hits: {cache_hits}, writes: {cache_writes}, directory: {source_cache_dir}"
        )
    
    # Aggregate results
    aggregated_results = aggregate_results(all_results)
    
    # Print results
    print_results(
        aggregated_results,
        source_dir=source_dir,
        translated_dir=translated_dir
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'semantic_consistency_results_{timestamp}.json'
    save_results(
        aggregated_results,
        output_path,
        detailed_results=all_results if args.save_detailed else None,
        source_dir=source_dir,
        translated_dir=translated_dir
    )
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
