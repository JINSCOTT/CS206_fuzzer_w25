import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Performing multiple math operations
        addition = x + 2
        subtraction = x - 2
        multiplication = x * 2
        division = x / 2
        
        # Performing comparison operations
        greater_than = x > 1
        less_than = x < 3
        
        return addition, subtraction, multiplication, division, greater_than, less_than

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    torch.tensor([[0.5, 1.5], [2.5, 3.5]]),
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
    torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]], [[[5.0, 6.0]], [[7.0, 8.0]]]])
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        results = model(tensor)
        print(f"Input: {tensor}\nOutput: {results}\n")