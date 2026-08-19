# Machine Learning: Linear Regression with Gradient Descent

## Project Overview

This project implements a **linear regression model** to predict house prices based on square footage (sqft_living) using the **gradient descent optimization algorithm**. The project demonstrates both vectorized and non-vectorized implementations to help understand how gradient descent works mathematically.

---

## File Structure

```
├── main.py                 # Vectorized implementation (optimized)
├── gradient_descent.py     # Non-vectorized implementation (educational)
├── README.md              # Project documentation
└── docs/
    ├── API.md             # API reference
    └── USAGE.md           # Usage guide
```

---

## main.py - Optimized Linear Regression

### Purpose
Implements a complete linear regression model with **vectorized NumPy operations** for efficiency. This is the production-ready version.

### Key Components

**1. Data Loading & Normalization**
- Loads house price dataset from CSV
- Normalizes the feature (sqft_living) using z-score normalization:
  - Formula: `x_normalized = (x - mean) / std_dev`
  - Improves gradient descent convergence

**2. Cost Function**
```python
compute_cost(x, y, w, b)
```
- Calculates Mean Squared Error (MSE): `J(w,b) = (1/2m) * Σ(f_wb - y)²`
- Measures how well the model fits the training data
- Used to track training progress

**3. Gradient Calculation**
```python
dj_d_wb(x, y, w, b)
```
- Computes partial derivatives of the cost function:
  - `dj_dw = (1/m) * Σ((f_wb - y) * x)` → gradient w.r.t. weight
  - `dj_db = (1/m) * Σ(f_wb - y)` → gradient w.r.t. bias
- Uses **vectorized NumPy operations** for speed

**4. Gradient Descent Algorithm**
```python
gradient_descent(x, y, w=0, b=0, alpha=0.03, num_iters=5000)
```
- Iteratively updates weights and bias:
  - `w = w - alpha * dj_dw`
  - `b = b - alpha * dj_db`
- **Parameters**:
  - `alpha` = learning rate (controls step size)
  - `num_iters` = number of iterations
- Prints cost every 500 iterations to monitor convergence

**5. Model Training & Evaluation**
- Trains on the entire dataset with 10,000 iterations
- Makes predictions on test data (rows 1010-1020)
- Compares predictions vs. actual prices

**6. Visualization**
- Plots training data (blue scatter points)
- Overlays the fitted regression line (red line)
- Shows relationship between square footage and price

### Why Use This Version?
✅ **Fast** - Vectorized operations are ~100x faster than loops  
✅ **Scalable** - Works efficiently with large datasets  
✅ **Production-ready** - Optimal for real-world applications

---

## gradient_descent.py - Educational Implementation

### Purpose
Implements gradient descent using **explicit loops** (non-vectorized). This version prioritizes **clarity and understanding** over performance.

### Key Functions

**1. Gradient Calculation (Non-Vectorized)**
```python
dj_wb_cal(x, y, w, b)
```
- Loops through each data point individually
- Calculates partial derivatives step-by-step:
  ```python
  for i in range(m):
      fx_wb_i = w * x[i] + b                    # prediction
      dj_dw_i = (fx_wb_i - y[i]) * x[i]        # gradient contribution
      dj_db_i = (fx_wb_i - y[i])               # gradient contribution
      dj_dw += dj_dw_i
      dj_db += dj_db_i
  ```
- Easier to follow the mathematical logic

**2. Gradient Descent Algorithm**
```python
gradient_descent(x, y, w_init=0, b_init=0, alpha=0.01, num_iters=1000)
```
- Same optimization logic as main.py but implemented with loops
- Updates parameters iteratively
- Prints progress every 100 iterations

### Why Use This Version?
📚 **Educational** - Shows math behind gradient descent clearly  
🔍 **Debugging** - Easy to trace and understand each step  
💡 **Learning** - Perfect for understanding how algorithms work

### Why NOT in Production?
❌ **Slow** - Loops are ~100x slower than vectorized operations  
❌ **Not scalable** - Inefficient for large datasets

---

## How Gradient Descent Works

### Concept
Gradient descent is an optimization algorithm that finds the best weights (w) and bias (b) by:
1. Starting with random initial values
2. Computing gradients (direction of steepest descent)
3. Taking small steps in the opposite direction of the gradient
4. Repeating until convergence

### The Learning Rate (alpha)
- **Too small** (e.g., 0.001): Slow convergence, more iterations needed
- **Too large** (e.g., 0.1): May overshoot, diverge, or oscillate
- **Optimal** (e.g., 0.03): Fast, stable convergence

### Math Formula
```
w = w - alpha * (dJ/dw)
b = b - alpha * (dJ/db)
```

---

## Model Equation

The linear regression model is:
```
f(x) = w * x + b
```

Where:
- **w** = weight (slope of the line)
- **b** = bias (y-intercept)
- **x** = normalized square footage
- **f(x)** = predicted house price

---

## Usage

### Requirements
```
pandas
numpy
matplotlib
```

### Running the Main Model
```bash
python main.py
```

### Expected Output
- Prints cost, w, and b every 500 iterations
- Shows predictions for test data
- Displays scatter plot with regression line

---

## Dataset
- Located at: `C:/Users/nuesp/Desktop/PYTHON/linear_regression_one_feature/dataset.csv`
- Contains: square footage (sqft_living) and house prices
- Features: 1 (sqft_living)
- Format: CSV

---

## Key Concepts

| Concept | Definition |
|---------|-----------|
| **Gradient Descent** | Optimization algorithm to minimize cost function |
| **Cost Function** | MSE measures prediction error |
| **Normalization** | Scaling features to improve convergence |
| **Learning Rate** | Step size for parameter updates |
| **Vectorization** | Using NumPy arrays instead of loops for speed |
| **Convergence** | Point where algorithm stops improving |

---

## Comparison: main.py vs gradient_descent.py

| Aspect | main.py | gradient_descent.py |
|--------|---------|-------------------|
| Implementation | Vectorized (NumPy) | Non-vectorized (Loops) |
| Speed | Fast (~100x) | Slow |
| Readability | Concise | Explicit |
| Use Case | Production | Learning |
| Scalability | Excellent | Poor |

---

## Future Improvements
- [ ] Add multiple features (multivariate regression)
- [ ] Implement regularization (L1/L2)
- [ ] Add cross-validation
- [ ] Use scikit-learn for comparison
- [ ] Add error metrics (MAE, RMSE, R²)

---

## Author
Machine Learning Learning Project

## License
Open source
