import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Normal mathematical operations
        return (x * 2) + 3 - (x / 5) ** 2

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),      # 2D
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D
    torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]),  # 4D
    torch.tensor([[5.0], [10.0], [15.0]]),           # 2D
    torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])  # 4D
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input: {input_tensor}, Output: {output}")