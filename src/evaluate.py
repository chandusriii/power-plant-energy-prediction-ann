"""Evaluate the saved ANN checkpoint on the held-out test partition."""

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import load_and_validate_data, make_dataloaders
from src.model import PowerPlantANN

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    df = load_and_validate_data(ROOT / "data" / "powerplant_data.csv")
    _, _, _, _, (x_test, y_test) = make_dataloaders(df, batch_size=32, seed=42)

    checkpoint_path = ROOT / "models" / "best_model.pt"
    scaler_path = ROOT / "models" / "scaler.joblib"
    if not checkpoint_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError("Model artifacts are missing. Run `python -m src.train` first.")
    # make_dataloaders returns x_test after applying the training-fitted scaler.
    # Load the persisted scaler to ensure the trained inference bundle is complete,
    # but do not transform x_test a second time.
    joblib.load(scaler_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = PowerPlantANN(input_features=checkpoint["input_features"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        predictions = model(torch.tensor(x_test, dtype=torch.float32)).numpy().reshape(-1)

    actual = y_test.reshape(-1)
    mse = mean_squared_error(actual, predictions)
    metrics = {
        "mae": float(mean_absolute_error(actual, predictions)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predictions)),
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
