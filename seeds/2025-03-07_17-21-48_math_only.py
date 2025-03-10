import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = x + 2  # Addition
        x = x - 3  # Subtraction
        x = x * 4  # Multiplication
        x = x / 2  # Division
        x = x ** 2  # Exponentiation
        return x

# Input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),         # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),      # 4D tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),           # 2D tensor
    torch.tensor([1.0, 2.0, 3.0]),                     # 1D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]]])            # 3D integer tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input: {tensor}, Output: {output}")