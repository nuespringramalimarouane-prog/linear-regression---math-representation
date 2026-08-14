# Usage

This document shows how to run and tweak the example in `main.py`.

## Running

1. Activate your environment.
2. Ensure dependencies are installed (`uv add numpy`, `uv add matplotlib`).
3. Run the script:

```powershell
python main.py
```

The script prints previews of `x_train` and `y_train`, plots a scatter of the generated data, and computes model outputs via `compute_model_output()`.

## Changing generated data

The generator signature in `main.py`:

```
generate_synthetic_data(n=50, w=200.0, b=100.0, x_min=0.5, x_max=3.0, noise_std=20.0, seed=None)
```

- Increase `n` to create more samples.
- Change `w` and `b` to alter the underlying linear relationship.
- Adjust `noise_std` to control noise.
- Use `seed` for reproducible runs.
