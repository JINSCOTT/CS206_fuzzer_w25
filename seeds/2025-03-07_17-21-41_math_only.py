import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        y = x + 2  # Addition
        y = y * 3  # Multiplication
        y = y - 1  # Subtraction
        y = y / 4  # Division
        return y

# Input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),    # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1, 2]], [[3, 4]]]], dtype=torch.float32),     # 4D tensor
    torch.tensor([[0, -1, -2], [3, 4, 5]], dtype=torch.float32),   # 2D tensor with negative values
    torch.tensor([[[0.1], [0.2]], [[0.3], [0.4]]], dtype=torch.float32)  # 3D tensor with decimals
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')