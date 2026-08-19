import numpy as np

"""
this code here implements gradient descent mathimatically 
and this is a simple implementation of linear regression using gradient descent.
but i didn't use it because it's a little bit slower than the vectorized implementation in main.py, but it is easier to understand and implement.

dj_wb_cal : 
    This function calculates the gradients of the cost function with respect to the weights (w) and bias (b) for linear regression. 
    It takes in the input features (x), target values (y), current weight (w), and current bias (b) as parameters. 
    It returns the gradients dj_dw and dj_db.

gradient_descent :
    This function performs the gradient descent optimization algorithm to find the optimal values of weight (w)
    and bias (b) that minimize the cost function.


"""
def dj_wb_cal(x,y,w,b):
    m = len(x)
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        fx_wb_i = w * x[i] + b

        dj_dw_i = (fx_wb_i - y[i]) * x[i]

        dj_db_i = (fx_wb_i - y[i]) 

        dj_dw = dj_dw + dj_dw_i
        dj_db = dj_db + dj_db_i

    dj_db = (1/m) * dj_db
    dj_dw = (1/m) * dj_dw

    return dj_dw , dj_db


def gradient_descent(x, y, w_init=0, b_init=0, alpha=0.01, num_iters=1000):
    w = w_init
    b = b_init

    for i in range(num_iters):
        dj_dw, dj_db = dj_wb_cal(x, y, w, b)

        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Optional: print progress every 100 iterations
        if i % 100 == 0:
            cost = np.mean((w * x + b - y) ** 2)
            print(f"Iteration {i}: w={w:.4f}, b={b:.4f}, cost={cost:.4f}")

    return w, b
