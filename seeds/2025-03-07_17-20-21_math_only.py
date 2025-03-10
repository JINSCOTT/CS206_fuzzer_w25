import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        x = x + 2  # Addition
        x = x * 3  # Multiplication
        x = x - 1  # Subtraction
        x = x / 4  # Division
        return x

# Input Tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=torch.float32),
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
    torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32),
    torch.tensor([[[10], [20]], [[30], [40]], [[50], [60]]], dtype=torch.float32),
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output_tensor = model(input_tensor)
        print(f"Output for input tensor {i}:")
        print(output_tensor)