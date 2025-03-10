import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example of normal math operations
        x = x + 2  # Addition
        x = x - 1  # Subtraction
        x = x * 3  # Multiplication
        x = x / 4  # Division
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),
    torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32),
    torch.tensor([[1], [2], [3], [4]], dtype=torch.float32),
    torch.tensor([[[[1]]], [[[2]]], [[[3]]]], dtype=torch.float32),
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output_tensor = model(input_tensor)
        print(f"Input Tensor:\n{input_tensor}\nOutput Tensor:\n{output_tensor}\n")