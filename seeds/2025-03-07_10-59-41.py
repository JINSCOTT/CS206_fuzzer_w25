import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example math operations
        addition = x + 2
        subtraction = x - 2
        multiplication = x * 2
        division = x / 2

        # Example comparison operations
        is_greater = x > 1
        is_less_equal = x <= 3

        return addition, subtraction, multiplication, division, is_greater, is_less_equal

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], [[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]]),  # 4D tensor
    torch.tensor([[1.0], [2.0], [3.0]]),  # 2D tensor
    torch.tensor([1.0, 2.0, 3.0])  # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        result = model(tensor)
        print(result)