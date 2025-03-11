import torch
import torch.utils.benchmark as benchmark
import pandas as pd
import matplotlib.pyplot as plt

# Define tensor sizes for testing
tensor_sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000), (3000, 3000), (4000, 4000), (5000, 5000), (6000, 6000)]
results = []

# TorchInductor optimization setup
torch._dynamo.config.suppress_errors = True

def test_fn(x, y):
    return torch.add(x, y), x + y

optimized_fn = torch.compile(test_fn)  # Compile for TorchInductor

# Run tests for different tensor sizes
for size in tensor_sizes:
    a = torch.randn(size, dtype=torch.float16, device="cpu")  # Use half precision
    b = torch.randn(size, dtype=torch.float16, device="cpu")

    # Measure execution times (eager mode) with fewer iterations
    t1 = benchmark.Timer(stmt="torch.add(a, b)", globals={"a": a, "b": b, "torch": torch}).timeit(50)
    t2 = benchmark.Timer(stmt="a + b", globals={"a": a, "b": b}).timeit(50)

    # Run optimized version (TorchInductor)
    out_eager = test_fn(a, b)
    out_optimized = optimized_fn(a, b)

    # Ensure results are the same
    assert torch.allclose(out_eager[0], out_optimized[0]), "Torch.add mismatch in Inductor!"
    assert torch.allclose(out_eager[1], out_optimized[1]), "+ operator mismatch in Inductor!"

    # Store results
    results.append((size, t1, t2))

# Create DataFrame for results
df = pd.DataFrame(results, columns=["Tensor Size", "torch.add Time (s)", "+ Operator Time (s)"])
# Extract execution times and print them in seconds
df["torch.add Time (s)"] = df["torch.add Time (s)"].apply(lambda x: x.mean)
df["+ Operator Time (s)"] = df["+ Operator Time (s)"].apply(lambda x: x.mean)

print("\nExecution Time Results (in seconds):")
print(df.to_string(index=False))  # Print DataFrame without index for readability

# Plot results with spacing adjustments
plt.figure(figsize=(10, 6))  # Increase figure size
plt.plot([str(s) for s in tensor_sizes], [r[1].mean for r in results], marker='o', label="torch.add")
plt.plot([str(s) for s in tensor_sizes], [r[2].mean for r in results], marker='s', label="+ Operator")

plt.xlabel("Tensor Size")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time Comparison: torch.add vs + Operator")
plt.legend()
plt.grid(True)

# Rotate x-axis labels and add padding
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()  # Ensures everything fits nicely

plt.show()
