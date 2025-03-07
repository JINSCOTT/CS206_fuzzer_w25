import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Normal math operations
        x = x + 5              # Addition
        x = x - 2              # Subtraction
        x = x * 3              # Multiplication
        x = x / 4              # Division
        x = x ** 2             # Exponentiation
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),               # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]]]),           # 4D tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),             # 2D tensor with floats
    torch.tensor([[[10]], [[20]], [[30]]])              # 3D tensor with a single dimension
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {i+1}:\n{output}")