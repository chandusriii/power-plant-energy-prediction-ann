# Power Plant Energy Prediction with an Artificial Neural Network

A reproducible deep-learning regression project built with **PyTorch**. The model predicts electrical energy output (`PE`) from four environmental measurements: ambient temperature (`AT`), exhaust vacuum (`V`), ambient pressure (`AP`), and relative humidity (`RH`).

> Training and evaluation were run locally on the documented 70/15/15 split (seed 42). The held-out test metrics are shown below.

## Project objective

Learn a nonlinear mapping from power-plant environmental conditions to electrical energy output using a feed-forward Artificial Neural Network (ANN).

## Dataset

The CSV dataset is expected at `data/powerplant_data.csv`.

| Column | Meaning | Role |
|---|---|---|
| `AT` | Ambient temperature | Input |
| `V` | Exhaust vacuum | Input |
| `AP` | Ambient pressure | Input |
| `RH` | Relative humidity | Input |
| `PE` | Electrical energy output | Target |

The pipeline validates required columns, missing values, and numeric data types before training.

## Model architecture

The baseline network follows the architecture in the original notebook:

```text
4 input features → Dense(6) + ReLU → Dense(6) + ReLU → Dense(1)
```

- Framework: PyTorch
- Hidden activation: ReLU
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Batch size: 32
- Default epochs: 100

The output layer is linear for continuous regression.

## Project structure

```text
power-plant-energy-prediction-ann/
├── data/
│   └── powerplant_data.csv
├── notebooks/
│   └── ANN_Regression_original.ipynb
├── src/
│   ├── model.py
│   ├── data.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   └── test_model.py
├── models/                 # generated model artifacts
├── results/                # generated metrics and plots
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/chandusriii/power-plant-energy-prediction-ann.git
cd power-plant-energy-prediction-ann

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train

Run from the repository root:

```bash
python -m src.train
```

Optional settings:

```bash
python -m src.train --epochs 100 --batch-size 32 --learning-rate 0.001 --seed 42
```

The data is split into training, validation, and test partitions. The scaler is fitted on the training partition only. Validation loss is used for checkpoint selection; the test set is reserved for final evaluation.

Generated outputs:
- `models/best_model.pt`
- `models/scaler.joblib`
- `results/training_history.csv`
- `results/loss_curve.png`
- `results/test_metrics.json`

## Evaluate

After training:

```bash
python -m src.evaluate
```

The evaluation reports MAE, MSE, RMSE, and R². Error metrics are in the target variable's original units; R² is unitless.

## Run tests

```bash
pytest -q
```


## Verified baseline results

The documented 100-epoch run (batch size 32, Adam learning rate 0.001, seed 42) produced the following held-out test metrics:

| Metric | Result |
|---|---:|
| MAE | 3.5762 |
| MSE | 20.4877 |
| RMSE | 4.5263 |
| R² | 0.9299 |

Metrics are from the test partition after selecting the checkpoint by validation MSE. Results can vary with the environment and dependency versions.

## Key implementation choices

1. **Leakage-aware preprocessing:** `StandardScaler` is fitted on training data only.
2. **Three-way split:** validation data supports model selection; test data is reserved for final evaluation.
3. **Reproducibility:** a configurable random seed is used for data splitting and model initialization.
4. **Best checkpoint:** the model with the lowest validation MSE is saved.
5. **Readable outputs:** training history, a loss plot, and final test metrics are saved separately.

## Learning outcomes

- Preparing tabular data for deep learning
- Building a fully connected ANN in PyTorch
- Understanding forward propagation and backpropagation
- Training with mini-batches and Adam
- Monitoring validation loss and selecting a checkpoint
- Evaluating regression models with standard metrics

## Scope and future improvements

This is an educational minor project and a baseline ANN implementation. Results depend on the data split, random seed, and training configuration. Future work could compare the ANN with linear regression, random forest, and gradient boosting; tune hyperparameters; and study feature importance.

## License

Released under the MIT License. See [LICENSE](LICENSE).
