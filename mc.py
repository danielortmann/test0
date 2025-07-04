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
