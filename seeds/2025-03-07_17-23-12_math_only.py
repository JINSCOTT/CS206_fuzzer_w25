import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Normal math operations
        x = x + 2  # Addition
        x = x - 1  # Subtraction
        x = x * 3  # Multiplication
        x = x / 4  # Division
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.float32),  # 2D tensor
    torch.tensor([1, 2, 3], dtype=torch.float32),  # 1D tensor
    torch.tensor([[[1]], [[2]], [[3]]], dtype=torch.float32)  # 4D tensor with single elements
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {i}: {output}")