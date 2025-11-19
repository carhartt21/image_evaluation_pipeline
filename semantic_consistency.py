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
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DeepLabV3Evaluator:
    """
    Evaluator for semantic consistency using DeepLabV3 segmentation.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet101',
        device: Optional[str] = None,
        num_classes: int = 20  # Cityscapes has 19 classes + background
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
        
        print(f"Initializing DeepLabV3 with {backbone} backbone on {self.device}...")
        
        # Load pre-trained DeepLabV3 model
        if backbone == 'resnet50':
            self.model = models.segmentation.deeplabv3_resnet50(
                pretrained=False,
                progress=True
            )
            self.model.load_state_dict(torch.load('weights/deeplabv3_resnet50_cityscapes.bin'))
        elif backbone == 'resnet101':
            self.model = models.segmentation.deeplabv3_resnet101(
                pretrained=True,
                progress=True
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
    def compute_iou(mask1: np.ndarray, mask2: np.ndarray, num_classes: int) -> Dict[str, float]:
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
                class_ious[f'class_{cls}'] = iou
        
        # Compute mean IoU
        miou = np.mean(ious) if ious else 0.0
        
        return {
            'mIoU': miou,
            'class_IoUs': class_ious
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
        iou_metrics = self.compute_iou(source_mask, translated_mask, self.num_classes)
        
        return {
            'pixel_accuracy': pixel_acc,
            'mIoU': iou_metrics['mIoU'] * 100.0,  # Convert to percentage
            'class_IoUs': iou_metrics['class_IoUs']
        }


def get_image_pairs(
    source_dir: Path,
    translated_dir: Path,
    extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp'),
    strip_suffixes: Tuple[str, ...] = ()
) -> List[Tuple[Path, Path]]:
    """
    Find matching image pairs in source and translated directories.
    
    Args:
        source_dir: Directory with source images
        translated_dir: Directory with translated images
        extensions: Allowed image file extensions
        strip_suffixes: Filename suffixes to drop before matching
        
    Returns:
        List of (source_path, translated_path) tuples
    """
    def normalize_stem(stem: str) -> str:
        """Strip configured suffixes from a filename stem."""
        if not strip_suffixes:
            return stem
        for suffix in strip_suffixes:
            if stem.endswith(suffix):
                return stem[:-len(suffix)]
        return stem
    
    pairs = []
    
    # Get all source images
    source_images = [
        f for f in source_dir.iterdir()
        if f.suffix.lower() in extensions
    ]
    
    # Build lookup for translated images using normalized stems
    translated_lookup: Dict[str, List[Path]] = {}
    for translated_path in translated_dir.iterdir():
        if translated_path.suffix.lower() not in extensions:
            continue
        key = normalize_stem(translated_path.stem)
        translated_lookup.setdefault(key, []).append(translated_path)
    
    for source_path in source_images:
        key = normalize_stem(source_path.stem)
        candidates = translated_lookup.get(key)
        if not candidates:
            print(f"Warning: No matching translated image for {source_path.name}")
            continue
        # Prefer translated image with matching extension if available
        match = next(
            (candidate for candidate in candidates
             if candidate.suffix.lower() == source_path.suffix.lower()),
            candidates[0]
        )
        pairs.append((source_path, match))
    
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
    detailed_results: Optional[List[Dict[str, Any]]] = None
):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Aggregated results dictionary
        output_path: Path to output JSON file
        detailed_results: Optional list of per-pair results
    """
    output_data = {
        'summary': results
    }
    
    if detailed_results:
        output_data['detailed_results'] = detailed_results
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_results(results: Dict[str, Any]):
    """
    Print formatted evaluation results.
    
    Args:
        results: Aggregated results dictionary
    """
    print("\n" + "="*60)
    print("SEMANTIC CONSISTENCY EVALUATION RESULTS")
    print("="*60)
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
        description='Evaluate semantic consistency in image-to-image translation using DeepLabV3',
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
        default='resnet101',
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
    
    args = parser.parse_args()
    
    # Convert paths
    source_dir = Path(args.source_dir)
    translated_dir = Path(args.translated_dir)
    output_dir = Path(args.output_dir)
    
    # Validate directories
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not translated_dir.exists():
        raise FileNotFoundError(f"Translated directory not found: {translated_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = DeepLabV3Evaluator(
        backbone=args.backbone,
        device=args.device
    )
    
    # Get image pairs
    print("Finding image pairs...")
    image_pairs = get_image_pairs(
        source_dir,
        translated_dir,
        extensions=tuple(args.extensions),
        strip_suffixes=tuple(args.strip_suffixes) if args.strip_suffixes else ()
    )
    
    if not image_pairs:
        print("Error: No matching image pairs found!")
        return
    
    print(f"Found {len(image_pairs)} image pairs\n")
    
    # Evaluate all pairs
    all_results = []
    
    print("Evaluating semantic consistency...")
    for source_path, translated_path in tqdm(image_pairs, desc="Processing"):
        try:
            result = evaluator.evaluate_pair(source_path, translated_path)
            result['source_image'] = source_path.name
            result['translated_image'] = translated_path.name
            all_results.append(result)
        except Exception as e:
            print(f"\nError processing {source_path.name}: {str(e)}")
            continue
    
    # Aggregate results
    aggregated_results = aggregate_results(all_results)
    
    # Print results
    print_results(aggregated_results)
    
    # Save results
    output_path = output_dir / 'semantic_consistency_results.json'
    save_results(
        aggregated_results,
        output_path,
        detailed_results=all_results if args.save_detailed else None
    )
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
