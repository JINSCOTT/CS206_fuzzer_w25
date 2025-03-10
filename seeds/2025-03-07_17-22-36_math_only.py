import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example of normal math operations
        x = x + 2  # Addition
        x = x - 3  # Subtraction
        x = x * 4  # Multiplication
        x = x / 2  # Division
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),                     # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]]]),                 # 4D tensor
    torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]]),  # 2D tensor
    torch.tensor([[[[1, 2]], [[3, 4]], [[5, 6]]]])           # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print("Input:\n", tensor)
        print("Output:\n", output)