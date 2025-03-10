import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
    
    def forward(self, inputs):
        a, b, c, d, e = inputs
        
        # Math operations
        add_result = a + b
        sub_result = b - c
        mul_result = c * d
        div_result = d / (e + 1e-5)  # Avoid division by zero
        
        # Comparison operations
        greater_than = a > b
        less_than = b < c
        equal_to = c == d
        
        return add_result, sub_result, mul_result, div_result, greater_than, less_than, equal_to

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[5.0], [6.0]], [[7.0], [8.0]]]),  # 3D tensor
    torch.tensor([1.0]),  # 1D tensor
    torch.tensor([[10.0, 20.0]]),  # 2D tensor
    torch.tensor([0.0])  # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    results = model(input_tensors)
    for result in results:
        print(result)