import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Performing some ordinary math operations
        x = x + 2  # Addition
        x = x * 3  # Multiplication
        x = x - 1  # Subtraction
        x = x / 2  # Division
        return x

# Defining input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[2, 4], [6, 8], [10, 12]], dtype=torch.float32)  # 2D tensor
]

if __name__ == "__main__":
    module = PtModule()
    for tensor in input_tensors:
        output = module(tensor)
        print(output)