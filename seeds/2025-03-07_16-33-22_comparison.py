import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = {}
        
        # Math operations
        results['add'] = inputs[0] + inputs[1]
        results['subtract'] = inputs[2] - inputs[3]
        results['multiply'] = inputs[0] * inputs[4]
        results['divide'] = inputs[1] / (inputs[2] + 1e-5)  # Adding small value to prevent division by zero
        
        # Comparison operations
        results['greater_than'] = inputs[0] > inputs[1]
        results['less_than'] = inputs[2] < inputs[3]
        results['equal'] = inputs[4] == inputs[0]
        
        return results

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[0, 1], [1, 0]])  # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    
    for key, value in outputs.items():
        print(f"{key}: {value}")