# API

This project is intentionally small. Public helpers are in `main.py` and `src/machinelearning`.

## `generate_synthetic_data(...)`

Generates synthetic linear-regression data.

Parameters:

- `n` (int): number of samples
- `w` (float): slope
- `b` (float): intercept
- `x_min`, `x_max` (float): range for `x`
- `noise_std` (float): standard deviation of Gaussian noise
- `seed` (int|None): RNG seed for reproducibility

Returns: `(x, y)` NumPy arrays of shape `(n,)`.

## `compute_model_output(x, w, b)`

Computes the prediction `f_wb = w*x + b` for each element in `x`.

Parameters:

- `x` (ndarray): input array shape `(m,)`
- `w`, `b` (scalars): model parameters

Returns: `f_wb` (ndarray shape `(m,)`).
