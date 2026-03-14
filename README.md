# Linear Regression with PyTorch

A simple linear regression implementation using PyTorch, demonstrating how to build, train, and evaluate a basic neural network model from scratch. The model learns the relationship `y = 2x + 1` on a small synthetic dataset.

## Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3 | Programming language |
| 🔥 PyTorch | Deep learning framework |
| 🔢 NumPy | Numerical computing / data prep |

## Dependencies

```
torch
numpy
```

Install with:

```bash
pip install torch numpy
```

## How to Run

```bash
python linear-regression.py
```

The script will train a linear regression model for 100 epochs using SGD and print the loss at each epoch. GPU acceleration is used automatically if CUDA is available; otherwise it falls back to CPU.

## What It Does

1. Generates a synthetic dataset based on `y = 2x + 1`
2. Defines a single-layer linear model (`nn.Linear`)
3. Trains using Mean Squared Error loss and Stochastic Gradient Descent
4. Logs the training loss per epoch

## Known Issues

- The dataset is hardcoded and very small (11 samples) — this is intentional for demonstration purposes.
- No validation/test split or evaluation metrics beyond training loss.
- No model saving or inference script included.

## License

See [LICENSE](LICENSE) for details.
