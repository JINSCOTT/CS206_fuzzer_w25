import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Perform basic mathematical operations
        x = x + 2       # Addition
        x = x - 3       # Subtraction
        x = x * 4       # Multiplication
        x = x / 5       # Division
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),  # 3D Tensor
    torch.tensor([[2, 3], [5, 7]], dtype=torch.float32),                                    # 2D Tensor
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32),              # 3D Tensor
    torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32),                  # 2D Tensor
    torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)                                     # 1D Tensor
]

if __name__ == "__main__":
    model = PtModule()
    for i, tensor in enumerate(input_tensors):
        output = model(tensor)
        print(f"Output for input tensor {i}:\n{output}")