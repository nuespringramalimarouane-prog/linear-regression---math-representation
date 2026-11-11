# Linear Regression with Gradient Descent

A small educational project that implements linear regression with NumPy and compares two approaches to gradient descent:

- **Vectorized implementation** in `main.py`.
- **Loop-based implementation** in `gradient_descent.py`.

The project focuses on the prediction function, mean-squared-error cost, gradient calculation, parameter updates, and training-cost visualization.

## Concepts covered

- Linear regression: \(f_{w,b}(x) = wx + b\)
- Mean-squared-error cost
- Partial derivatives with respect to the weight and bias
- Batch gradient descent
- Feature normalization
- Vectorized NumPy operations versus explicit Python loops

## Repository structure

```text
.
├── main.py                 # Vectorized implementation and visualization
├── gradient_descent.py     # Loop-based educational implementation
├── README.md
└── docs/
    ├── API.md
    └── USAGE.md
```

The program also expects the dataset CSV used by the implementation. Keep the dataset in the location configured by the data-loading code, or update that path for your environment.

## Requirements

- Python 3
- NumPy
- pandas
- Matplotlib

Install the dependencies:

```bash
python -m pip install numpy pandas matplotlib
```

## Run

From the repository directory, run:

```bash
python main.py
```

The script trains the vectorized model, reports training progress and predictions, and displays a plot of the data and fitted regression line. The exact output depends on the dataset and the parameters configured in the source code.

## How the implementation works

### Prediction

For one feature, the model predicts:

\[
f_{w,b}(x) = wx + b
\]

where \(w\) is the weight, \(b\) is the bias, and \(x\) is the normalized input feature.

### Cost function

The implementation uses the half mean-squared-error cost:

\[
J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)})-y^{(i)})^2
\]

The cost measures the average squared difference between predictions and actual values. The factor of \(1/2\) simplifies the gradient expressions.

### Gradient descent

At each iteration, the algorithm computes the gradients and updates the parameters in the opposite direction:

\[
w := w - \alpha\frac{\partial J}{\partial w},
\qquad
b := b - \alpha\frac{\partial J}{\partial b}
\]

The learning rate \(\alpha\) controls the size of each update. The implementation records or prints cost values to help inspect convergence.

## Implementation comparison

| Aspect | `main.py` | `gradient_descent.py` |
|---|---|---|
| Approach | Vectorized NumPy operations | Explicit Python loops |
| Main goal | Demonstrate array-based computation | Make each calculation step visible |
| Readability | More concise | More step-by-step |
| Learning value | Efficient numerical implementation | Easier to trace individual updates |

The vectorized implementation is intended to reduce Python-level looping; actual performance depends on the dataset size and execution environment.

## Current limitations

- The training and evaluation configuration is defined in the source code rather than exposed as command-line options.
- The evaluation slice is small, so its metrics should not be treated as a reliable estimate of generalization.
- The dataset path must be configured for the local environment.
- The implementation is educational and is not presented as a production forecasting system.

## Possible next improvements

- Use a reproducible train/validation/test split.
- Evaluate on a larger held-out test set.
- Report MAE, RMSE, and \(R^2\).
- Add tests for prediction, cost, gradient, and parameter updates.
- Handle missing values and constant features explicitly.
- Add a `requirements.txt` file and a configurable dataset path.
