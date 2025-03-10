import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Normal math operations
        y = x + 2  # Addition
        y = y * 3  # Multiplication
        y = y - 1  # Subtraction
        y = y / 4  # Division
        return y

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0]], [[[2.0]], [[3.0]], [[4.0]]]]]),  # 4D tensor
    torch.tensor([1.0, 2.0, 3.0]),  # 1D tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")