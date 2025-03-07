import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Basic mathematical operations
        x = x + 2          # Addition
        x = x - 1          # Subtraction
        x = x * 3          # Multiplication
        x = x / 4          # Division
        x = x ** 2         # Exponentiation
        x = x % 5          # Modulus
        return x

input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
    torch.tensor([[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]]),
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    torch.tensor([[1.0], [2.0], [3.0]]),
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")