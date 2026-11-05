"""Train the ANN and save the best checkpoint selected by validation MSE."""

import argparse
import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import load_and_prepare_data
from src.model import PowerPlantANN

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, loss_function, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss, total_samples = 0.0, 0
    for features, targets in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predictions = model(features)
            loss = loss_function(predictions, targets)
            if training:
                loss.backward()
                optimizer.step()
        count = features.size(0)
        total_loss += loss.item() * count
        total_samples += count
    return total_loss / max(total_samples, 1)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    torch.set_num_threads(1)
    data = load_and_prepare_data(ROOT / "data" / "powerplant_data.csv", args.batch_size, args.seed)
    model = PowerPlantANN()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    models_dir, results_dir = ROOT / "models", ROOT / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    best_validation_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, data["train_loader"], loss_function, optimizer)
        validation_loss = run_epoch(model, data["validation_loader"], loss_function)
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": validation_loss})
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": [4, 6, 6, 1],
                "input_features": ["AT", "V", "AP", "RH"],
                "best_validation_mse": best_validation_loss,
                "epoch": epoch, "seed": args.seed,
            }, models_dir / "best_model.pt")
        print(f"Epoch {epoch:03d}/{args.epochs} | train MSE: {train_loss:.5f} | validation MSE: {validation_loss:.5f}")

    joblib.dump(data["scaler"], models_dir / "scaler.joblib")
    pd.DataFrame(history).to_csv(results_dir / "training_history.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot([row["train_mse"] for row in history], label="Training MSE")
    plt.plot([row["validation_mse"] for row in history], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("ANN Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "loss_curve.png", dpi=150)
    plt.close()
    summary = {
        "best_validation_mse": best_validation_loss,
        "best_epoch": min(history, key=lambda row: row["validation_mse"])["epoch"],
        "split_sizes": data["split_sizes"], "epochs_requested": args.epochs, "seed": args.seed,
    }
    (results_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTraining complete.\n" + json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
