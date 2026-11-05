"""Evaluate the saved ANN checkpoint on the held-out test partition."""

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import load_and_prepare_data
from src.model import PowerPlantANN

ROOT = Path(__file__).resolve().parents[1]


def evaluate() -> dict[str, float]:
    torch.set_num_threads(1)
    checkpoint_path = ROOT / "models" / "best_model.pt"
    scaler_path = ROOT / "models" / "scaler.joblib"
    if not checkpoint_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError("Model artifacts are missing. Run `python -m src.train` first.")

    data = load_and_prepare_data(ROOT / "data" / "powerplant_data.csv", batch_size=256, random_state=42)
    scaler = joblib.load(scaler_path)
    model = PowerPlantANN()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # Reconstruct raw test features using the deterministic train-fitted scaler,
    # then transform with the persisted scaler used by the saved model.
    raw_x_test = data["scaler"].inverse_transform(data["x_test"])
    x_test = scaler.transform(raw_x_test).astype(np.float32)
    with torch.no_grad():
        predictions = model(torch.from_numpy(x_test)).numpy().reshape(-1)
    actual = data["y_test"].reshape(-1)
    mse = mean_squared_error(actual, predictions)
    metrics = {
        "mae": float(mean_absolute_error(actual, predictions)),
        "mse": float(mse), "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predictions)),
    }
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
