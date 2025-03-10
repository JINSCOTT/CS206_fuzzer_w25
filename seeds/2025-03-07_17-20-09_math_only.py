import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Sample normal math operations
        x = x + 2          # Addition
        x = x - 1          # Subtraction
        x = x * 3          # Multiplication
        x = x / 4          # Division
        x = x ** 2         # Exponentiation
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32),               # 3D tensor
    torch.tensor([[[[5, 6], [7, 8]]]], dtype=torch.float32),           # 4D tensor
    torch.tensor([[9, 10], [11, 12]], dtype=torch.float32),             # 2D tensor
    torch.tensor([[[13], [14]], [[15], [16]]], dtype=torch.float32),    # 4D tensor
    torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)                 # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output_tensor = model(input_tensor)
        print(f"Input: {input_tensor}\nOutput: {output_tensor}\n")