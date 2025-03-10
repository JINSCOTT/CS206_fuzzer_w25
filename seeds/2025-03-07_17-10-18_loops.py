import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        results = {}
        
        # Addition
        results['addition'] = x + 2
        
        # Subtraction
        results['subtraction'] = x - 2
        
        # Multiplication
        results['multiplication'] = x * 2
        
        # Division
        results['division'] = x / 2
        
        # Looping through the tensor to compute the square
        squared_values = torch.empty_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                squared_values[i][j] = x[i][j] ** 2
        
        results['squared'] = squared_values
        
        return results

# Input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),  # 2D tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),  # 2D tensor
    torch.tensor([1.0, 2.0, 3.0])  # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input tensor:\n{input_tensor}\nOutput:\n{output}\n")