import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""

This code implements a simple linear regression model using gradient descent to predict house prices based on square footage (sqft_living).

"""

# Load dataset
df = pd.read_csv("dataset.csv")

# Normalize only the feature (sqft_living)
x_mean, x_std = df["sqft_living"].mean(), df["sqft_living"].std()
x_train = (df["sqft_living"] - x_mean) / x_std
y_train = df["price"]

# compute gradients of the cost function with respect to weights and bias
def dj_d_wb(x, y, w, b):
    m = len(x)
    f_wb = w * x + b
    dj_dw = np.sum((f_wb - y) * x) / m
    dj_db = np.sum(f_wb - y) / m
    return dj_dw, dj_db
# compute cost function
def compute_cost(x, y, w, b):
    m = len(x)
    f_wb = w * x + b
    return np.sum((f_wb - y)**2) / (2*m)
# gradient descent function
def gradient_descent(x, y, w=0, b=0, alpha=0.03, num_iters=5000):
    for i in range(num_iters):
        dj_dw, dj_db = dj_d_wb(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i % 500 == 0:
            print(f"Iteration {i}: Cost {compute_cost(x,y,w,b)}, w={w}, b={b}")
    return w, b

# Train model
w, b = gradient_descent(x_train.values, y_train.values, alpha=0.001, num_iters=10000)
print(f"Final parameters: w={w}, b={b}")

# Test data (normalize x only)
data_test = df["sqft_living"][1010:1020]
data_test_y = df["price"][1010:1020]
x_test = (data_test - x_mean) / x_std

# Predictions
for i in range(len(x_test)):
    prediction = w * x_test.iloc[i] + b
    print(f"Prediction for sqft_living {data_test.iloc[i]}: {prediction:.2f} Actual price: {data_test_y.iloc[i]}")

# Plot regression line in original scale
plt.scatter(df["sqft_living"], df["price"], color='blue', label='Training Data')
plt.plot(df["sqft_living"], w * ((df["sqft_living"] - x_mean)/x_std) + b, color='red', label='Regression Line')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('Linear Regression: Price vs Square Footage')
plt.legend()
plt.show()
