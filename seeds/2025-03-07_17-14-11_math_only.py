import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Normal math operations
        x = x + 2        # Addition
        x = x - 3        # Subtraction
        x = x * 4        # Multiplication
        x = x / 5        # Division
        x = torch.pow(x, 2)  # Power
        return x

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),  # 2D Tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D Tensor
    torch.tensor([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]], dtype=torch.float32),  # 4D Tensor
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),  # 2D Tensor
    torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]], dtype=torch.float32)  # 4D Tensor
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {i}: {output}")