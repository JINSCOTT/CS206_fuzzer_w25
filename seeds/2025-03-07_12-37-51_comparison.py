import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Mathematical operations
        addition = x + 5
        subtraction = x - 2
        multiplication = x * 3
        division = x / 4

        # Comparison operations
        greater_than = x > 1
        less_than = x < 10
        equal_to = x == 3

        return addition, subtraction, multiplication, division, greater_than, less_than, equal_to

# Input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    torch.tensor([[10, 20, 30], [40, 50, 60]]),
    torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    torch.tensor([[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]]]),
    torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        result = model(tensor)
        print(result)