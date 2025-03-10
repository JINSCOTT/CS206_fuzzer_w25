import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        addition = x + 2
        subtraction = x - 2
        multiplication = x * 2
        division = x / 2
        comparison = x > 1
        return addition, subtraction, multiplication, division, comparison

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),    # 2D
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D
    torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=torch.float32),  # 4D
    torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),  # 2D
    torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32)  # 3D
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        result = model(input_tensor)
        print(result)