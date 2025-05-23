import torch
import time
import numpy as np
import matplotlib.pyplot as plt

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create random tensors
size = (1000, 1000)  # Large tensor for benchmarking
A = torch.randn(size, device=device)
B = torch.randn(size, device=device)
C = torch.randn(size, device=device)
D = torch.randn(size, device=device)

# Function to measure execution time over multiple runs
def benchmark(func, label, runs=10000, discard=100):
    times = []
    for _ in range(runs):
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        func()
        torch.cuda.synchronize() if device == "cuda" else None
        end_time = time.time()
        times.append(end_time - start_time)

    # Remove warm-up runs
    stable_times = times[discard:]

    # Compute mean and std deviation
    avg_time = np.mean(stable_times)
    std_time = np.std(stable_times)
    print(f"{label}: {avg_time:.6f} ± {std_time:.6f} seconds")

    return avg_time

# Define test cases
test_cases = {
    "A + B - C": lambda: A + B - C,
    "torch.add(A, B) - C": lambda: torch.add(A, B) - C,
    "A * B / C": lambda: A * B / C,
    "torch.mul(A, B) / C": lambda: torch.mul(A, B) / C,
    "A + B * C - A / B": lambda: A + B * C - A / B,
    "torch.add(A, torch.mul(B, C)) - torch.div(A, B)": lambda: torch.add(A, torch.mul(B, C)) - torch.div(A, B),
    "(A - B) * (C + D) / A": lambda: (A - B) * (C + D) / A,
    "torch.mul(torch.sub(A, B), torch.div(torch.add(C, D), A))": lambda: torch.mul(torch.sub(A, B), torch.div(torch.add(C, D), A)),
    "(A * B) + (C / D) - (A / B)": lambda: (A * B) + (C / D) - (A / B),
    "torch.add(torch.mul(A, B), torch.sub(torch.div(C, D), torch.div(A, B)))": lambda: torch.add(torch.mul(A, B), torch.sub(torch.div(C, D), torch.div(A, B)))
}

# Run benchmarks and store results
results = {}
for label, func in test_cases.items():
    results[label] = benchmark(func, label, runs=10000, discard=100)

# Convert results to lists for plotting
labels = list(results.keys())
times = list(results.values())

# Plot the execution times
plt.figure(figsize=(12, 6))
plt.barh(labels, times, color='skyblue')
plt.xlabel("Execution Time (seconds)")
plt.ylabel("Operations")
plt.title("Execution Time for Different Operation Combinations (Averaged Over 10,000 Runs)")
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.grid(True)
plt.show()
