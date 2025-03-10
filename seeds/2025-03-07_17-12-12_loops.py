import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Using mathematical operations
        x = x + 2  # Addition
        x = x * 3  # Multiplication
        x = x - 5  # Subtraction
        x = x / 2  # Division

        # Performing a loop
        for i in range(2):
            x = x + i  # Adding the loop index

        return x

# Input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),  # 2D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),  # 4D tensor
    torch.tensor([[2.0, 3.0], [4.0, 5.0]]),  # 2D tensor with floats
    torch.tensor([[[[1]]], [[[2]]], [[[3]]]])  # 4D tensor with single values
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input: {tensor}\nOutput: {output}\n")