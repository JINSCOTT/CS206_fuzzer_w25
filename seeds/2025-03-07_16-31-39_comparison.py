import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2):
        # Arithmetic operations
        addition = x1 + x2
        subtraction = x1 - x2
        multiplication = x1 * x2
        division = x1 / (x2 + 1e-5)  # Prevent division by zero

        # Comparison operations
        greater_than = x1 > x2
        less_than = x1 < x2
        equal_to = x1 == x2
        not_equal_to = x1 != x2

        return {
            'addition': addition,
            'subtraction': subtraction,
            'multiplication': multiplication,
            'division': division,
            'greater_than': greater_than,
            'less_than': less_than,
            'equal_to': equal_to,
            'not_equal_to': not_equal_to
        }

# Defining input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),           # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),         # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),       # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),           # 2D tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])          # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    results = model(input_tensors[0], input_tensors[3])
    print(results)