import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = {}
        a, b, c = inputs
        
        # Addition
        results['add'] = a + b
        
        # Subtraction
        results['sub'] = a - b
        
        # Multiplication
        results['mul'] = a * b
        
        # Division
        results['div'] = a / (b + 1e-10)  # Avoid division by zero
        
        # Comparisons
        results['greater'] = a > c
        results['less'] = a < c
        results['equal'] = a == c
        
        return results

input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D Tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),  # 2D Tensor
    torch.tensor([[2.0, 3.0], [4.0, 5.0]])   # 2D Tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)