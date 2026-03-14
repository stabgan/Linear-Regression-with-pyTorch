# Linear Regression with PyTorch

A minimal PyTorch example that trains a single-layer linear model on a toy dataset (`y = 2x + 1`) using stochastic gradient descent.

## What It Does

1. Generates a small synthetic dataset of 11 points following `y = 2x + 1`.
2. Trains an `nn.Linear` layer with MSE loss and SGD for 100 epochs.
3. Prints the learned weight and bias, plus per-sample predictions.

The script auto-detects CUDA and falls back to CPU gracefully.

## 🛠 Tech Stack

| Component | Tool |
|-----------|------|
| 🧠 Framework | [PyTorch](https://pytorch.org/) |
| 🔢 Numerics | [NumPy](https://numpy.org/) |
| 🐍 Language | Python 3.8+ |

## Getting Started

```bash
# Install dependencies
pip install torch numpy

# Run
python linear-regression.py
```

### Expected Output

```
Using device: cpu
Epoch [  1/100], Loss: 91.5731
Epoch [ 10/100], Loss: 0.1553
...
Learned parameters:
  linear.weight: [~2.0]
  linear.bias:   [~1.0]
```

## ⚠️ Known Issues

- Dataset is hardcoded — useful only as a learning example.
- No train/test split (toy dataset, not needed here).

## License

MIT
