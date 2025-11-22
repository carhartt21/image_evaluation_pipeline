"""Tests for semantic consistency integration helpers."""
from pathlib import Path
from typing import List

import pytest
import torch

from ..evaluate_generation import compute_semantic_consistency
from ..utils.image_io import LoadedImagePair


class DummyEvaluator:
    """Simple stand-in for SegFormerEvaluator used during testing."""

    def __init__(self, results: List[dict]):
        self._results = results
        self._idx = 0

    def evaluate_pair(self, source_path: Path, translated_path: Path) -> dict:
        if self._idx >= len(self._results):
            raise AssertionError("More evaluate_pair calls than expected")
        result = {
            **self._results[self._idx],
            "source_image": str(source_path),
            "translated_image": str(translated_path),
        }
        self._idx += 1
        return result


def test_compute_semantic_consistency_aggregates_scores():
    pairs = [
        LoadedImagePair(
            gen_tensor=torch.zeros(3, 4, 4),
            real_tensor=torch.zeros(3, 4, 4),
            name="sample_a",
            gen_path=Path("/tmp/gen_a.png"),
            real_path=Path("/tmp/real_a.png"),
        ),
        LoadedImagePair(
            gen_tensor=torch.ones(3, 4, 4),
            real_tensor=torch.ones(3, 4, 4),
            name="sample_b",
            gen_path=Path("/tmp/gen_b.png"),
            real_path=Path("/tmp/real_b.png"),
        ),
    ]

    dummy_results = [
        {
            "pixel_accuracy": 80.0,
            "mIoU": 60.0,
            "fw_IoU": 65.0,
            "class_IoUs": {"road": 0.5, "sky": 0.7},
            "class_details": {"road": {"IoU": 0.5, "frequency": 0.2}},
        },
        {
            "pixel_accuracy": 90.0,
            "mIoU": 70.0,
            "fw_IoU": 75.0,
            "class_IoUs": {"road": 0.6},
            "class_details": {"road": {"IoU": 0.6, "frequency": 0.3}},
        },
    ]

    def factory(**_: object):
        return DummyEvaluator(dummy_results)

    payload = compute_semantic_consistency(
        pairs=pairs,
        model_variant="segformer-b0",
        device="cpu",
        show_progress=False,
        evaluator_factory=factory,
    )

    scalars = payload["scalars"]
    metadata = payload["metadata"]
    summary = metadata["summary"]

    assert scalars["sample_a"]["semantic_pixel_accuracy"] == pytest.approx(80.0)
    assert scalars["sample_b"]["semantic_mIoU"] == pytest.approx(70.0)

    assert summary["num_pairs_evaluated"] == 2
    assert summary["average_pixel_accuracy"] == pytest.approx(85.0)
    assert summary["average_mIoU"] == pytest.approx(65.0)
    assert summary["average_fw_IoU"] == pytest.approx(70.0)
    assert summary["average_class_IoUs"]["road"] == pytest.approx(55.0)
    assert summary["average_class_IoUs"]["sky"] == pytest.approx(70.0)
    road_details = summary["average_fw_class_details"]["road"]
    assert road_details["average_IoU"] == pytest.approx(55.0)
    assert road_details["average_frequency"] == pytest.approx(25.0)

    assert metadata["model_variant"] == "segformer-b0"
    assert metadata["model_name"].endswith("cityscapes-768-768")


def test_compute_semantic_consistency_handles_empty_pairs():
    flag = {"called": False}

    def factory(**_: object):
        flag["called"] = True
        return DummyEvaluator([])

    payload = compute_semantic_consistency(
        pairs=[],
        model_variant="segformer-b1",
        device="cpu",
        show_progress=False,
        evaluator_factory=factory,
    )

    assert payload["scalars"] == {}
    assert payload["metadata"]["enabled"] is False
    assert payload["metadata"]["summary"] == {}
    assert flag["called"] is False, "Factory should not be used for empty input"
