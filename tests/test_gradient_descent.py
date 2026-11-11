import numpy as np

from gradient_descent import dj_wb_cal, gradient_descent


def test_gradient_calculation_on_small_dataset():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])

    dw, db = dj_wb_cal(x, y, w=0.0, b=0.0)

    assert np.isclose(dw, -28.0 / 3.0)
    assert np.isclose(db, -4.0)


def test_gradient_descent_reduces_prediction_error():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])

    w, b = gradient_descent(x, y, alpha=0.05, num_iters=500)

    assert np.mean((w * x + b - y) ** 2) < 0.01
