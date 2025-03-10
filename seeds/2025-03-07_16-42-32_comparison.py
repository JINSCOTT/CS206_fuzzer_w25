import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Applying various mathematical operations
        addition = x + 5
        subtraction = x - 2
        multiplication = x * 3
        division = x / 4
        power = x ** 2

        # Applying some comparison operations
        greater_than = x > 10
        less_than = x < 5
        equal_to = x == 3

        return {
            'addition': addition,
            'subtraction': subtraction,
            'multiplication': multiplication,
            'division': division,
            'power': power,
            'greater_than': greater_than,
            'less_than': less_than,
            'equal_to': equal_to
        }

# Input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([10, 11, 12, 13], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32),
    torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()

    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")