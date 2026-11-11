from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"


def load_dataset(path=DATASET_PATH):
    df = pd.read_csv(path)
    X = df["sqft_living"].to_numpy(dtype=float)
    y = df["price"].to_numpy(dtype=float)
    return X, y


def split_data(X, y, train_size=15000):
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if not 0 < train_size < len(X):
        raise ValueError("train_size must be between 0 and the sample count.")

    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]


def normalize_features(X_train, X_test):
    mean = X_train.mean()
    std = X_train.std()
    if std == 0:
        raise ValueError("Cannot normalize a constant feature.")
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def predict(x, w, b):
    return w * x + b


def compute_cost(x, y, w, b):
    errors = predict(x, w, b) - y
    return np.sum(errors ** 2) / (2 * len(x))


def compute_gradients(x, y, w, b):
    errors = predict(x, w, b) - y
    return np.sum(errors * x) / len(x), np.sum(errors) / len(x)


def gradient_descent(x, y, w=0.0, b=0.0, alpha=0.01, num_iters=1000):
    cost_history = []
    for iteration in range(num_iters):
        dj_dw, dj_db = compute_gradients(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost_history.append(compute_cost(x, y, w, b))
        if iteration % 100 == 0:
            print(
                f"Iteration {iteration:4d} | Cost = {cost_history[-1]:.2f} "
                f"| w = {w:.4f} | b = {b:.4f}"
            )
    return w, b, cost_history


def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        raise ValueError("R² is undefined when the target is constant.")
    return 1 - (ss_res / ss_tot)


def main():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_norm, X_test_norm, x_mean, x_std = normalize_features(X_train, X_test)

    w, b, cost_history = gradient_descent(
        X_train_norm, y_train, alpha=0.01, num_iters=1000
    )
    y_test_pred = predict(X_test_norm, w, b)
    y_train_pred = predict(X_train_norm, w, b)

    print("\nFinal parameters:")
    print(f"w = {w}")
    print(f"b = {b}")
    print(f"R² train = {r2_score_np(y_train, y_train_pred):.4f}")
    print(f"R² test = {r2_score_np(y_test, y_test_pred):.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("J(w, b)")
    plt.title("Cost Function Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    x_line = np.linspace(X.min(), X.max(), 200)
    y_line = predict((x_line - x_mean) / x_std, w, b)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, label="Training Data")
    plt.plot(x_line, y_line, label="Regression Line")
    plt.xlabel("Square Footage")
    plt.ylabel("Price")
    plt.title("Linear Regression: Price vs Square Footage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
