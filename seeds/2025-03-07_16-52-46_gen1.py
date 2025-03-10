import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensors):
        results = []
        for tensor in input_tensors:
            # Simple math operations
            processed_tensor = tensor * 2 + 1  # Example operation
            results.append(processed_tensor)
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[1, 0, 1], [0, 1, 0]], [[1, 1, 1], [0, 0, 0]]], dtype=torch.float32),
    torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4], [5, 6]]], dtype=torch.float32),
    torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    for i, out in enumerate(output):
        print(f"Output tensor {i}: {out}")