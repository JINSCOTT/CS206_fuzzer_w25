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
        comparison_gt = x > 1
        comparison_lt = x < 1
        comparison_eq = x == 2
        
        return {
            'addition': addition,
            'subtraction': subtraction,
            'multiplication': multiplication,
            'division': division,
            'comparison_gt': comparison_gt,
            'comparison_lt': comparison_lt,
            'comparison_eq': comparison_eq
        }

input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),
    torch.tensor([3.0, 4.0, 5.0])
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)