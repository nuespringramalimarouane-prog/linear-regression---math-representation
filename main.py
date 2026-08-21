import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "C:/Users/nuesp/Desktop/PYTHON/linear_regression_one _feature/dataset.csv"
)

X = df["sqft_living"].to_numpy(dtype=float)
y = df["price"].to_numpy(dtype=float)


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================
# For now, we use the same indices you were using.
# Later, we can replace this with train_test_split.

X_train = X[:15000]
y_train = y[:15000]

X_test = X[15000:]
y_test = y[15000:]


# ============================================================
# 3. FEATURE NORMALIZATION
# ============================================================
# IMPORTANT:
# Calculate mean and std ONLY from the training data.

x_mean = X_train.mean()
x_std = X_train.std()

X_train_norm = (X_train - x_mean) / x_std
X_test_norm = (X_test - x_mean) / x_std


# ============================================================
# 4. LINEAR REGRESSION MODEL
# ============================================================
# Model:
#
#       f(x) = wx + b
#
# ============================================================


def predict(x, w, b):
    """
    Calculate predictions.

    f(x) = wx + b
    """
    return w * x + b


# ============================================================
# 5. COST FUNCTION
# ============================================================

def compute_cost(x, y, w, b):
    """
    Mean Squared Error / 2

             1
    J = ----------- Σ(f(x) - y)²
             2m
    """

    m = len(x)

    predictions = predict(x, w, b)

    errors = predictions - y

    cost = np.sum(errors ** 2) / (2 * m)

    return cost


# ============================================================
# 6. GRADIENT
# ============================================================

def compute_gradients(x, y, w, b):
    """
    Calculate the gradients:

              1
    dj_dw = ----- Σ(f(x) - y)x
              m

              1
    dj_db = ----- Σ(f(x) - y)
              m
    """

    m = len(x)

    predictions = predict(x, w, b)

    errors = predictions - y

    dj_dw = np.sum(errors * x) / m

    dj_db = np.sum(errors) / m

    return dj_dw, dj_db


# ============================================================
# 7. GRADIENT DESCENT
# ============================================================

def gradient_descent(
    x,
    y,
    w=0.0,
    b=0.0,
    alpha=0.1,
    num_iters=1000
):

    cost_history = []

    for i in range(num_iters):

        # --------------------------------
        # Calculate gradients
        # --------------------------------

        dj_dw, dj_db = compute_gradients(
            x,
            y,
            w,
            b
        )

        # --------------------------------
        # Update parameters
        # --------------------------------

        w -= alpha * dj_dw

        b -= alpha * dj_db

        # --------------------------------
        # Calculate cost
        # --------------------------------

        cost = compute_cost(
            x,
            y,
            w,
            b
        )

        cost_history.append(cost)

        # --------------------------------
        # Display progress
        # --------------------------------

        if i % 100 == 0:

            print(
                f"Iteration {i:4d} | "
                f"Cost = {cost:.2f} | "
                f"w = {w:.4f} | "
                f"b = {b:.4f}"
            )

    return w, b, cost_history


# ============================================================
# 8. TRAIN MODEL
# ============================================================

w, b, cost_history = gradient_descent(
    X_train_norm,
    y_train,
    alpha=0.01,
    num_iters=1000
)

print("\nFinal parameters:")
print(f"w = {w}")
print(f"b = {b}")


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

y_pred = predict(
    X_test_norm,
    w,
    b
)

y_train_pred = predict(
    X_train_norm,
    w,
    b
)

# ============================================================
# 10. DISPLAY PREDICTIONS
# ============================================================

print("\nPredictions:")

for i in range(len(X_test)):

    print(
        f"sqft_living = {X_test[i]:.0f} | "
        f"Predicted = {y_pred[i]:.2f} | "
        f"Actual = {y_test[i]:.2f}"
    )


# ============================================================
# 11. R² SCORE
# ============================================================

def r2_score_np(y_true, y_pred):

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    return 1 - (ss_res / ss_tot)


r2 = r2_score_np(
    y_test,
    y_pred
)

r2_train = r2_score_np(y_train,y_train_pred)

print(f"\nR² train = {r2_train:.4f}")
print(f"\nR² test = {r2:.4f}")


# ============================================================
# 12. COST FUNCTION CONVERGENCE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(len(cost_history)),
    cost_history
)

plt.xlabel("Iterations")
plt.ylabel("J(w, b)")
plt.title("Cost Function Convergence")

plt.grid(True)

plt.show()


# ============================================================
# 13. REGRESSION LINE
# ============================================================

# Generate many x values for a smooth regression line.

x_line = np.linspace(
    X.min(),
    X.max(),
    200
)

# Normalize using training statistics.

x_line_norm = (
    x_line - x_mean
) / x_std

# Predict prices.

y_line = predict(
    x_line_norm,
    w,
    b
)


plt.figure(figsize=(8, 5))

plt.scatter(
    X_train,
    y_train,
    label="Training Data"
)

plt.plot(
    x_line,
    y_line,
    label="Regression Line"
)

plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("Linear Regression: Price vs Square Footage")

plt.legend()
plt.grid(True)

plt.show()