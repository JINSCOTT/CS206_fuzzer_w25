import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = {}
        a, b, c, d, e = inputs

        results['add'] = a + b
        results['subtract'] = a - b
        results['multiply'] = a * b
        results['divide'] = a / (b + 1e-6)  # Adding a small value to avoid division by zero
        results['power'] = a ** 2
        results['equal'] = a == b
        results['greater_than'] = a > b
        results['less_than'] = a < b
        
        return results

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[5.0, 4.0], [3.0, 2.0]]),  # 2D tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),  # 4D tensor
    torch.tensor([[5, 5], [5, 5]])  # 2D integer tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)