import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Apply various mathematical operations
        addition = x + 2
        subtraction = x - 1
        multiplication = x * 3
        division = x / 2
        power = x ** 2

        # Apply comparisons
        greater_than = x > 1
        less_than = x < 5
        equal_to = x == 3

        return addition, subtraction, multiplication, division, power, greater_than, less_than, equal_to


input_tensors = [
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
    torch.tensor([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], [[1.0, 0.0, -1.0], [-2.0, -3.0, -4.0]]]),
    torch.tensor([[[1.0, 0.0, 2.0]], [[3.0, 4.0, 5.0]], [[6.0, 7.0, 8.0]]]),
    torch.tensor([[[10.0], [20.0]], [[30.0], [40.0]]]),
    torch.tensor([[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]])
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        result = model(input_tensor)
        print(f"Input {i+1} results: {result}")