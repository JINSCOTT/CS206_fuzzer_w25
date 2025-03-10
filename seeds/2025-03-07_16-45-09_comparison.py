import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Math operations
        addition = x + 10
        subtraction = x - 5
        multiplication = x * 2
        division = x / 2
        
        # Comparison operations
        greater_than = x > 5
        less_than = x < 5
        equal_to = x == 5

        return {
            'addition': addition,
            'subtraction': subtraction,
            'multiplication': multiplication,
            'division': division,
            'greater_than': greater_than,
            'less_than': less_than,
            'equal_to': equal_to
        }

# Input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[2, 3, 4], [5, 6, 7]]], dtype=torch.float32),
    torch.tensor([[[1], [2], [3], [4]]], dtype=torch.float32),
    torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32),
    torch.tensor([[[10]], [[20]], [[30]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)