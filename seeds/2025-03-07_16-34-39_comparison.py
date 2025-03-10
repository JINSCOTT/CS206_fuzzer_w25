import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = {}

        # Addition
        results['add'] = inputs[0] + inputs[1]
        
        # Subtraction
        results['subtract'] = inputs[0] - inputs[1]
        
        # Multiplication
        results['multiply'] = inputs[0] * inputs[1]
        
        # Division
        results['divide'] = inputs[0] / (inputs[1] + 1e-5)  # Added epsilon to avoid division by zero
        
        # Greater than
        results['greater_than'] = inputs[0] > inputs[1]
        
        # Less than
        results['less_than'] = inputs[0] < inputs[1]
        
        return results

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D Tensor
    torch.tensor([[0.5, 1.5], [2.5, 3.5]]),  # 2D Tensor
    torch.tensor([[[1, 2], [3, 4]]]),          # 3D Tensor
    torch.tensor([[[[1, 2], [3, 4]]]]),        # 4D Tensor
    torch.tensor([[1, 0], [0, 1]])             # 2D Tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)