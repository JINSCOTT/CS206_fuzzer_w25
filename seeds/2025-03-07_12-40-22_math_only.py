import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        x = x + 5       # addition
        x = x - 2       # subtraction
        x = x * 3       # multiplication
        x = x / 4       # division
        x = x ** 2      # exponentiation
        return x

# Input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32),      # 3D tensor
    torch.tensor([[[[1, 2]], [[3, 4]]]], dtype=torch.float32), # 4D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1]], [[2]], [[3]]], dtype=torch.float32),    # 4D tensor
    torch.tensor([[10, 20], [30, 40]], dtype=torch.float32)     # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")