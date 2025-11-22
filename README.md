# PRISM: Pipeline for Robust Image Similarity Metrics

PRISM (Pipeline for Robust Image Similarity Metrics) is a comprehensive Python evaluation pipeline for comparing generated images against real images to assess the quality and realism of image-to-image translation or weather synthesis models.

## Features

- **Multiple Image Quality Metrics**: FID, SSIM, LPIPS, PSNR, Inception Score
- **Semantic Consistency Analysis**: Optional SegFormer-based evaluation for class-level agreement
- **Flexible Input Handling**: Support for different image formats and automatic pairing
- **Modular Architecture**: Easy addition of new metrics through plugin-style design
- **Batch Processing**: Efficient handling of large image datasets with progress tracking
- **Statistical Analysis**: Confidence intervals, significance testing, and distribution analysis
- **GPU Acceleration**: CUDA support for faster processing
- **Comprehensive Reporting**: JSON output with detailed per-image and summary statistics

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for faster processing)

### Setup

1. **Clone or extract the project:**
   ```bash
   cd image_evaluation_pipeline
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python evaluate_generation.py --help
   ```

## Quick Start

### Basic Usage

Evaluate generated images against reference images:

```bash
python evaluate_generation.py \
  --generated ./path/to/generated_images \
  --real ./path/to/reference_images \
  --metrics fid ssim lpips psnr \
  --output results.json

# Include semantic consistency
python evaluate_generation.py \
  --generated ./path/to/generated_images \
  --real ./path/to/reference_images \
  --metrics fid psnr \
  --semantic-consistency \
  --semantic-model segformer-b3 \
  --output results_with_semantic.json
```

### Weather Image Translation Example

For weather-specific evaluation:

```bash
python evaluate_generation.py \
  --generated ./generated_weather \
  --real ./real_weather \
  --metrics fid ssim lpips psnr is \
  --config configs/weather_eval.yaml \
  --batch-size 64 \
  --output weather_results.json \
  -v
```

## Usage Guide

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--generated` | Path to generated images directory | Required |
| `--real` | Path to reference images directory | Required |
| `--metrics` | Metrics to compute | `[fid, ssim, lpips, psnr, is]` |
| `--batch-size` | Batch size for processing | `32` |
| `--device` | Computation device (`cpu`, `cuda`, `auto`) | `auto` |
| `--pairs` | Pairing strategy (`auto`, `csv`) | `auto` |
| `--manifest` | CSV file for custom pairing | None |
| `--output` | Output file path | `results.json` |
| `--config` | Configuration file (YAML/JSON) | None |
| `--verbose` | Verbosity level (`-v`, `-vv`) | `0` |
| `--semantic-consistency` | Enable SegFormer-based semantic metrics | `False` |
| `--semantic-model` | SegFormer backbone variant (`segformer-b0` … `segformer-b5`) | `segformer-b5` |
| `--semantic-device` | Device for semantic evaluator (`cpu`, `cuda`, `auto`) | `auto` |

### Configuration Files

You can use YAML or JSON configuration files to specify parameters:

**Example YAML config (configs/weather_eval.yaml):**
```yaml
generated: ./generated_weather
real: ./real_weather
metrics: [fid, ssim, lpips, psnr]
batch_size: 64
device: auto
weather_categories: [fog, rain, snow, night, clear, cloudy]
output: weather_results.json
verbose: 1
image_size: [299, 299]
```

### Image Pairing Strategies

#### Automatic Pairing (default)
Images are paired by filename (ignoring extension):
```
generated/
  ├── image_001.png
  ├── image_002.jpg
  └── image_003.png

real/
  ├── image_001.jpg
  ├── image_002.png
  └── image_003.tiff
```

#### CSV Manifest Pairing
Create a CSV file with custom pairing:
```csv
gen_path,real_path
generated/fog_001.png,real/clear_001.jpg
generated/rain_002.png,real/sunny_002.png
```

Then use:
```bash
python evaluate_generation.py \
  --pairs csv \
  --manifest custom_pairs.csv \
  --output results.json
```

## Supported Metrics

### Fréchet Inception Distance (FID)
- **Range**: [0, ∞) (lower is better)
- **Description**: Measures similarity between feature representations of generated and real images
- **Use case**: Overall image quality and realism assessment

### Structural Similarity Index (SSIM)
- **Range**: [-1, 1] (higher is better)
- **Description**: Compares structural information between images
- **Use case**: Pixel-level similarity assessment

### Learned Perceptual Image Patch Similarity (LPIPS)
- **Range**: [0, ∞) (lower is better)
- **Description**: Perceptual similarity using deep features
- **Use case**: Human-perception-aligned similarity measurement

### Peak Signal-to-Noise Ratio (PSNR)
- **Range**: [0, ∞) (higher is better)
- **Description**: Ratio between signal power and noise power
- **Use case**: Pixel-level reconstruction quality

### Inception Score (IS)
- **Range**: [1, ∞) (higher is better)
- **Description**: Measures quality and diversity of generated images
- **Use case**: Generation quality assessment (generated images only)

## Output Format

The evaluation results are saved in JSON format with the following structure:

```json
{
  "timestamp": "2025-07-31T14:30:00",
  "generated": "./generated_weather",
  "real": "./real_weather",
  "metrics": {
    "fid": {
      "count": 100,
      "mean": 15.23,
      "std": 2.45,
      "median": 14.8,
      "min": 10.2,
      "max": 22.1,
      "q25": 13.5,
      "q75": 16.9,
      "confidence_interval": {
        "alpha": 0.95,
        "lower": 14.75,
        "upper": 15.71
      },
      "normality_test": {
        "test_name": "Shapiro-Wilk",
        "statistic": 0.985,
        "p_value": 0.234,
        "is_normal": true
      }
    }
  },
  "per_image": {
    "image_001": {
      "fid": 14.5,
      "ssim": 0.82,
      "lpips": 0.15,
      "psnr": 28.3
    }
  }
}
```

## Adding Custom Metrics

The pipeline uses a plugin-based architecture for easy metric extension:

1. **Create a new metric file** in the `metrics/` directory:

```python
# metrics/my_metric.py
import torch

class Metric:
    name = "my_metric"

    def __init__(self, device="cpu"):
        self.device = device
        # Initialize your metric here

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: generated images (B, C, H, W)
        # y: real images (B, C, H, W)
        # Return: metric scores (B,)
        scores = []
        for i in range(x.size(0)):
            score = compute_your_metric(x[i], y[i])
            scores.append(score)
        return torch.stack(scores)
```

2. **The metric is automatically discovered** and can be used:

```bash
python evaluate_generation.py --metrics my_metric ssim fid
```

## Testing

Run the test suite to verify installation:

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_metrics.py -v

# Run with coverage (if installed)
pytest tests/ --cov=.
```

## Performance Optimization

### GPU Acceleration
- Automatic GPU detection with `--device auto`
- Force GPU usage with `--device cuda`
- Batch processing for efficient GPU utilization

### Memory Management
- Adjust `--batch-size` based on available GPU memory
- Typical values: 16-32 for 8GB GPU, 64-128 for 16GB+ GPU

### Large Dataset Processing
```bash
# Process large datasets efficiently
python evaluate_generation.py \
  --generated ./large_dataset/generated \
  --real ./large_dataset/real \
  --batch-size 128 \
  --device cuda \
  --metrics fid ssim \
  --output large_results.json
```

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Solution: Reduce batch size
python evaluate_generation.py --batch-size 16 --device cuda
```

**2. No images found**
```bash
# Check supported formats: PNG, JPG, JPEG, TIFF, BMP
ls ./your_image_directory/
```

**3. Filename pairing issues**
```bash
# Use verbose mode to see pairing warnings
python evaluate_generation.py -v --generated ./gen --real ./real
```

**4. Import errors**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Logging

Enable detailed logging for debugging:

```bash
# Basic verbose mode
python evaluate_generation.py -v

# Very verbose mode with line numbers
python evaluate_generation.py -vv
```

## API Usage

You can also use the pipeline programmatically:

```python
from pathlib import Path
from utils.image_io import load_and_pair_images
from utils.stats import summarise_metrics
from metrics import registry

# Load and pair images
pairs = load_and_pair_images(
    gen_dir=Path("./generated"),
    real_dir=Path("./real")
)

# Initialize metrics
metrics = registry.build(["fid", "ssim"], device="cuda")

# Evaluate
results = {}
for gen_tensor, real_tensor, name in pairs:
    for metric in metrics:
        score = metric(gen_tensor.unsqueeze(0), real_tensor.unsqueeze(0))
        results.setdefault(name, {})[metric.name] = float(score)

# Summarize
summary = summarise_metrics(results)
print(summary)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request

## License

This project is provided as-is for research and educational purposes.

## Citation

If you use PRISM in your research, please cite:

```bibtex
@software{prism_image_metrics,
  title={PRISM: Pipeline for Robust Image Similarity Metrics},
  author={Christoph Gerhardt},
  year={2025},
  url={https://github.com/your-repo/image-evaluation-pipeline}
}
```

## Acknowledgments

- Built with PyTorch and TorchMetrics
- LPIPS implementation from [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
- FID implementation based on [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid)
