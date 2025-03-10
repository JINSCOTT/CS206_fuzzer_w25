import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        addition = x + 2
        subtraction = x - 2
        multiplication = x * 3
        division = x / 2
        greater_than = x > 1
        less_than = x < 3

        return addition, subtraction, multiplication, division, greater_than, less_than

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
    torch.tensor([[[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]]),
    torch.tensor([[1.0], [2.0], [3.0]]),
    torch.tensor([1.0, 2.0, 3.0, 4.0]),
    torch.tensor([[[0.5], [1.5]], [[2.5], [3.5]]])
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        result = model(input_tensor)
        print(f"Input:\n{input_tensor}\nResults:\n{result}\n")