import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = {}
        
        # Addition
        result['addition'] = x + 2
        
        # Subtraction
        result['subtraction'] = x - 2
        
        # Multiplication
        result['multiplication'] = x * 2
        
        # Division
        result['division'] = x / 2
        
        # Comparisons
        result['greater_than'] = x > 1
        result['less_than'] = x < 1
        result['equal_to'] = x == 1
        
        return result

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),  # 4D tensor
    torch.tensor([[1.0, 0.0], [0.0, 1.0]]),  # Another 2D tensor
    torch.tensor([1.0, 2.0, 3.0, 4.0])  # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input Tensor: {tensor.numpy()}")
        print(f"Output: {output}")