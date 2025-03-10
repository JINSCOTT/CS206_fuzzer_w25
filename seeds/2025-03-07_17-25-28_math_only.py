import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        x = x + 2        # Addition
        x = x * 3        # Multiplication
        x = x - 1        # Subtraction
        x = x / 4        # Division
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32),
    torch.tensor([[[5, 6, 7], [8, 9, 10]]], dtype=torch.float32),
    torch.tensor([[[11, 12, 13], [14, 15, 16], [17, 18, 19]]], dtype=torch.float32),
    torch.tensor([[20, 21], [22, 23]], dtype=torch.float32),
    torch.tensor([[[24]], [[25]], [[26]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")