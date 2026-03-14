"""
Linear Regression with PyTorch
Trains a simple linear model (y = 2x + 1) using SGD and MSE loss.
"""

import torch
import torch.nn as nn
import numpy as np


class LinearRegressionModel(nn.Module):
    """Single-layer linear regression model."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def main():
    # ── Device configuration ────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Dataset (y = 2x + 1) ────────────────────────────────────────
    x_values = list(range(11))
    x_train = np.array(x_values, dtype=np.float32).reshape(-1, 1)

    y_values = [2 * i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32).reshape(-1, 1)

    # Convert to tensors and move to device
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)

    # ── Model, loss, optimizer ──────────────────────────────────────
    input_dim = 1
    output_dim = 1
    model = LinearRegressionModel(input_dim, output_dim).to(device)

    criterion = nn.MSELoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # ── Training loop ───────────────────────────────────────────────
    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:>3}/{epochs}], Loss: {loss.item():.4f}")

    # ── Evaluation ──────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        predicted = model(inputs).cpu().numpy()

    print("\nLearned parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.data.cpu().numpy().flatten()}")

    print(f"\nSample predictions (expected y = 2x + 1):")
    for x, y_pred, y_true in zip(x_values, predicted.flatten(), y_values):
        print(f"  x={x:>2}  predicted={y_pred:.2f}  actual={y_true}")


if __name__ == "__main__":
    main()
