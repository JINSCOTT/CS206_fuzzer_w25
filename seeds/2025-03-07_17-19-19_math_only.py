import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Normal math operations
        x = x + 2  # Addition
        x = x * 3  # Multiplication
        x = x - 1  # Subtraction
        x = x / 2  # Division
        return x

# Defining the input tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=torch.float32),
    torch.tensor([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]], dtype=torch.float32),
    torch.tensor([[[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print("Output for input tensor:\n", input_tensor)
        print("Result:\n", output)