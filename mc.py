import os
import json
import numpy as np

def save_simulation(data_dict, param_dict, run_dir):
    """
    Save simulation results and parameters to a dedicated folder.

    Args:
        data_dict (dict): Dictionary of NumPy arrays.
        param_dict (dict): Dictionary of parameters (JSON-serializable).
        run_dir (str): Directory for this simulation run.
    """
    os.makedirs(run_dir, exist_ok=True)

    data_path = os.path.join(run_dir, "data.npz")
    param_path = os.path.join(run_dir, "params.json")

    np.savez(data_path, **data_dict)

    with open(param_path, "w") as f:
        json.dump(param_dict, f, indent=4)

    print(f"Saved data to:    {data_path}")
    print(f"Saved params to:  {param_path}")
    return  # Explicitly return None


def load_simulation(run_dir):
    """
    Load simulation results and parameters from a run folder.

    Args:
        run_dir (str): Path to the folder containing 'data.npz' and 'params.json'.

    Returns:
        data_dict (dict): Dictionary of NumPy arrays.
        param_dict (dict): Dictionary of parameters.
    """
    data_path = os.path.join(run_dir, "data.npz")
    param_path = os.path.join(run_dir, "params.json")

    data = np.load(data_path)
    data_dict = {key: data[key] for key in data.files}

    with open(param_path, "r") as f:
        param_dict = json.load(f)

    return data_dict, param_dict



####################
import numpy as np
import matplotlib.pyplot as plt
import time

# === Replaceable components ===

def sample(n):
    """Sampling function for random variable X (replace with your distribution)."""
    return np.random.rand(n)  # Example: uniform distribution

def f(x):
    """Function to compute expectation of f(X)."""
    return x**2  # Example: f(x) = x²

# === Simulation settings ===
total_samples = 100_000
snapshot_points = np.logspace(3, 5, num=10, dtype=int)
min_repetitions = 10

def simulate_mc_incremental(sample_size):
    """Simulate MC expectation estimation using incremental mean with vectorized sample and f(x)."""
    x = sample(sample_size)      # Vectorized sampling
    y = f(x)                      # Vectorized function application

    mean = 0.0
    for i, val in enumerate(y, 1):  # Incremental update
        mean += (val - mean) / i
    return mean

# === Run simulation ===
mean_estimates = []
std_estimates = []
mean_walltimes = []

for n_samples in snapshot_points:
    reps = max(min_repetitions, total_samples // n_samples)

    estimates = np.zeros(reps)
    walltimes = np.zeros(reps)

    for i in range(reps):
        start = time.time()
        estimates[i] = simulate_mc_incremental(n_samples)
        walltimes[i] = time.time() - start

    mean_estimates.append(np.mean(estimates))
    std_estimates.append(np.std(estimates))
    mean_walltimes.append(np.mean(walltimes))

# === Convert to arrays ===
mean_estimates = np.array(mean_estimates)
std_estimates = np.array(std_estimates)
mean_walltimes = np.array(mean_walltimes)

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.plot(mean_walltimes, mean_estimates, label='Estimate E[f(X)]', marker='o')
plt.fill_between(mean_walltimes,
                 mean_estimates - std_estimates,
                 mean_estimates + std_estimates,
                 alpha=0.3,
                 label='±1 std dev
