import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        addition = x + 2
        subtraction = x - 2
        multiplication = x * 2
        division = x / 2
        power = x ** 2
        comparison = x > 1
        
        # Return the results as a dictionary
        return {
            'addition': addition,
            'subtraction': subtraction,
            'multiplication': multiplication,
            'division': division,
            'power': power,
            'comparison': comparison
        }

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),          # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),        # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),      # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),          # 2D tensor
    torch.tensor([[[4.0, 5.0], [6.0, 7.0]]])         # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input Tensor:\n{input_tensor}\nOutput:\n{output}\n")