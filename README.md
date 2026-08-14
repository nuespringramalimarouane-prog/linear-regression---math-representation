# machinelearning

Small example project that demonstrates generating synthetic linear regression data and a minimal training script.

## Layout

- `main.py` — example script that generates data, plots it, and computes model outputs.
- `src/` — package sources.
  - `machinelearning/__init__.py` — package entrypoint (`main()`)
  - `machinelearning/__main__.py` — allow `python -m machinelearning`
- `pyproject.toml` — project metadata and `machinelearning` script entrypoint.

## Requirements

- Python 3.14+
- Managed with `uv` (uv_build)
- Runtime dependencies: `numpy`, `matplotlib`

## Quick start

1. Create and activate your virtual environment (if you use one):

```powershell
# Windows PowerShell
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

2. Install runtime packages with `uv` (project uses `pyproject.toml`):

```powershell
uv init            # run once if needed
uv add numpy
uv add matplotlib
```

3. Run the example script:

```powershell
python main.py
# or run as package
python -m machinelearning
```

## About the data

`main.py` uses a helper `generate_synthetic_data(n, w, b, noise_std, seed)` to create samples from the line

$$y = w x + b + \text{noise}$$

Default parameters match the tiny example in the repository: `w=200`, `b=100` (so `x=1 -> y≈300`, `x=2 -> y≈500`).

## Docs

See the `docs/` folder for usage and API details.

## License

MIT — feel free to change as needed.
