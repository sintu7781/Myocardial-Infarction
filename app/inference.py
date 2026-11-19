from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from .model_architectures import MIResNet1DDetector, MILocalizationResNet1D
except ImportError:
    from model_architectures import MIResNet1DDetector, MILocalizationResNet1D

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TERRITORIES = [
    "Inferior", "Anterior", "Anteroseptal", "Anterolateral",
    "Lateral", "Posterior", "Inferoposterior",
]
LOCALIZATION_COLUMNS = [
    "loc_inferior", "loc_anterior", "loc_anteroseptal",
    "loc_anterolateral", "loc_lateral", "loc_posterior",
    "loc_inferoposterior",
]


def validate_signal(signal: np.ndarray) -> np.ndarray:
    """Validate and standardize one ECG to (12, 5000), without resampling."""
    x = np.asarray(signal, dtype=np.float32)
    if x.shape == (5000, 12):
        x = x.T
    if x.shape != (12, 5000):
        raise ValueError(f"Expected 12x5000 or 5000x12 ECG array; got {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("ECG contains NaN or infinite values.")
    return x


def load_models(project_root: Path, device: torch.device):
    """Load exact model definitions and checkpoints from the training notebooks."""
    checkpoint_dir = project_root / "checkpoints"
    detection_path = checkpoint_dir / "resnet1d_mi_best.pt"
    localization_path = checkpoint_dir / "mi_localization_resnet1d_best.pt"
    if not detection_path.is_file():
        raise FileNotFoundError(f"Detection checkpoint missing: {detection_path}")
    if not localization_path.is_file():
        raise FileNotFoundError(f"Localization checkpoint missing: {localization_path}")

    detector = MIResNet1DDetector().to(device)
    localizer = MILocalizationResNet1D(num_classes=7).to(device)

    detection_ckpt = torch.load(detection_path, map_location=device, weights_only=True)
    localization_ckpt = torch.load(localization_path, map_location=device, weights_only=True)
    detector.load_state_dict(detection_ckpt["model_state_dict"], strict=True)
    localizer.load_state_dict(localization_ckpt["model_state_dict"], strict=True)
    detector.eval()
    localizer.eval()
    return detector, localizer, detection_ckpt, localization_ckpt


def load_thresholds(project_root: Path) -> tuple[float, np.ndarray]:
    """Load thresholds selected on validation data by the notebooks."""
    detection_metrics_path = project_root / "results" / "resnet1d" / "test_metrics.json"
    localization_threshold_path = (
        project_root / "results" / "localization_resnet1d" / "validation_thresholds.csv"
    )
    if not detection_metrics_path.is_file():
        raise FileNotFoundError(
            f"Missing {detection_metrics_path}; run detection evaluation cell first."
        )
    if not localization_threshold_path.is_file():
        raise FileNotFoundError(
            f"Missing {localization_threshold_path}; run localization threshold-selection cell first."
        )

    detection_metrics = json.loads(detection_metrics_path.read_text(encoding="utf-8"))
    detection_threshold = float(detection_metrics["threshold"])
    threshold_df = pd.read_csv(localization_threshold_path)

    threshold_col = "validation_selected_threshold"
    if threshold_col not in threshold_df or "territory" not in threshold_df:
        raise ValueError("Localization threshold CSV has an unexpected schema.")

    threshold_map = dict(zip(threshold_df["territory"], threshold_df[threshold_col]))
    thresholds = np.array(
        [float(threshold_map[column]) for column in LOCALIZATION_COLUMNS],
        dtype=np.float32,
    )
    return detection_threshold, thresholds


@torch.inference_mode()
def predict_ecg(
    signal_12x5000: np.ndarray,
    detector: torch.nn.Module,
    localizer: torch.nn.Module,
    device: torch.device,
    detection_threshold: float,
    localization_thresholds: np.ndarray,
) -> dict[str, Any]:
    """Return real model outputs, using the distinct training-time preprocessing."""
    raw = validate_signal(signal_12x5000)

    # Detection notebook fed raw WFDB physical-unit values (no z-score).
    x_detection = torch.from_numpy(raw).unsqueeze(0).to(device)

    # Localization notebook applied independent per-lead z-score normalization.
    means = raw.mean(axis=1, keepdims=True)
    stds = np.maximum(raw.std(axis=1, keepdims=True), 1e-6)
    normalized = (raw - means) / stds
    x_localization = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0).to(device)

    detection_probability = torch.sigmoid(detector(x_detection)).item()
    localization_probabilities = torch.sigmoid(localizer(x_localization)).cpu().numpy()[0]
    detection_label = detection_probability >= detection_threshold
    territory_results = [
        {
            "territory": territory,
            "probability": float(probability),
            "threshold": float(threshold),
            "positive": bool(probability >= threshold),
        }
        for territory, probability, threshold in zip(
            TERRITORIES, localization_probabilities, localization_thresholds
        )
    ]
    return {
        "mi_probability": float(detection_probability),
        "mi_threshold": float(detection_threshold),
        "mi_prediction": "MI-related pattern detected" if detection_label else "No MI-related pattern detected by model",
        "mi_positive": bool(detection_label),
        "territories": territory_results,
    }
