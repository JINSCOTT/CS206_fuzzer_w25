import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Normal math operations
        y = x + 2               # Addition
        y = y * 3               # Multiplication
        y = y - 5               # Subtraction
        y = y / 4               # Division
        return y

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),               # 2D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]]]),             # 4D tensor
    torch.tensor([[5, 6, 7], [8, 9, 10]]),                # Another 2D tensor
    torch.tensor([1.0, 2.0, 3.0])                         # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {i}:\n{output}")