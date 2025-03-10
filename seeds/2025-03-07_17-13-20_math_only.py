import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        x = x + 1          # Addition
        x = x - 2          # Subtraction
        x = x * 3          # Multiplication
        x = x / 4          # Division
        return x

# Input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]]]),             # 3D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6]]),         # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]]]),       # 3D tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),       # 2D tensor
    torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]]) # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input Tensor:\n{tensor}\nOutput Tensor:\n{output}\n")