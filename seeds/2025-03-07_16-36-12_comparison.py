import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Applying various math operations
        addition = x + 5
        subtraction = x - 3
        multiplication = x * 2
        division = x / 4
        power = x ** 2
        
        # Comparison operations
        greater_than = x > 2
        less_than = x < 10
        equal_to = x == 5
        
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

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D Tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),  # 3D Tensor
    torch.tensor([[[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]]),  # 4D Tensor
    torch.tensor([[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]),  # 2D Tensor
    torch.tensor([[[15.0, 16.0, 17.0]]])  # 3D Tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)