"""Dataset loading, splitting, scaling, and DataLoader creation."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

FEATURE_COLUMNS = ["AT", "V", "AP", "RH"]
TARGET_COLUMN = "PE"


def load_and_prepare_data(csv_path: str | Path, batch_size: int = 32, random_state: int = 42) -> dict[str, Any]:
    """Load the data and create 70/15/15 train/validation/test partitions."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame[required].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns.")
    try:
        values = frame[required].to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Required columns must be numeric.") from exc
    if not np.isfinite(values).all():
        raise ValueError("Required columns contain non-finite values.")

    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = frame[TARGET_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)
    x_train, x_temp, y_train, y_temp = train_test_split(features, targets, test_size=0.30, random_state=random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=random_state)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    def loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle)

    return {
        "train_loader": loader(x_train, y_train, True),
        "validation_loader": loader(x_val, y_val, False),
        "test_loader": loader(x_test, y_test, False),
        "scaler": scaler, "x_test": x_test, "y_test": y_test,
        "split_sizes": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
    }
