import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Perform normal math operations
        x = x + 2       # Addition
        x = x - 3       # Subtraction
        x = x * 4       # Multiplication
        x = x / 2       # Division
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),     # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([10, 20, 30], dtype=torch.float32),               # 1D tensor
    torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)    # 2D tensor
]

# Main section to check if the script runs
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")