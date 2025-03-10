import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Normal math operations
        # Example operations: addition, subtraction, multiplication, division, and power
        y1 = x + 2  # Addition
        y2 = y1 - 1  # Subtraction
        y3 = y2 * 3  # Multiplication
        y4 = y3 / 4.0  # Division
        y5 = y4 ** 2  # Power
        return y5

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32),
    torch.tensor([[[[1, 2]], [[3, 4]]]], dtype=torch.float32),
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32),
    torch.tensor([[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {i+1}:\n{output}")