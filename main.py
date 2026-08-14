import numpy as np
import matplotlib.pyplot as plt

def generate_london_prices(n=100, seed=None):
    """
    Generate simplified house size and price data for London.

    Args:
        n: number of samples
        seed: optional RNG seed

    Returns:
        x (ndarray (n,)): house sizes in 1000 sqft
        y (ndarray (n,)): house prices in GBP
    """
    rng = np.random.default_rng(seed)

    # House sizes between 0.5k and 5k sqft
    x = rng.uniform(0.5, 5.0, size=n)

    # Approximate average London house price (2026 estimate)
    base_price = 600_000  

    # Assume price grows with size (linear trend)
    price_per_sqft = 120_000  # GBP per 1000 sqft

    # Add variation to simulate different properties
    noise = rng.normal(0, 100_000, size=n)  # ±100k variation
    y = base_price + price_per_sqft * x + noise

    return x, y


def compute_model_output(x, w, b):
    return w * x + b


def compute_real_estate_prices(x, w, b):
    return w * x + b


def main():
    # Generate training data
    x_train, y_train = generate_london_prices(n=50, seed=0)

    # Fit linear regression (find w and b)
    w, b = np.polyfit(x_train, y_train, 1)
    print(f"Fitted parameters: w = {w:.2f}, b = {b:.2f}")

    # Predictions on training data
    f_wb = compute_model_output(x_train, w, b)

    
    # Test with new house sizes
    x_test = np.array([1.2, 2.4, 3.5])  # in 1000 sqft
    real_estimate = compute_real_estate_prices(x_test, w, b)
    print(f"Real estate price estimates for {x_test}: {real_estimate}")

    # Plot training data and fitted line
    plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
    plt.plot(x_train, f_wb, c='b', label='Fitted Model')
    plt.plot(x_test, real_estimate, marker='o', c='g', label='Test Predictions')
    plt.title("London Housing Prices")
    plt.xlabel("Size (1000 sqft)")
    plt.ylabel("Price (GBP)")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
