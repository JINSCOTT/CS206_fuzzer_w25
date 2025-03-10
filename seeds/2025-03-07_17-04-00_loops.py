import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
    
    def forward(self, x):
        # Example operations
        output = x * 2          # Multiplication
        output = output + 3     # Addition
        output = output - 1     # Subtraction
        output = output / 2      # Division
        
        # Loop through the dimensions and apply a basic operation
        for i in range(output.size(0)):
            for j in range(output.size(1)):
                output[i, j] = output[i, j] ** 2  # Squaring each element
        
        return output

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[0, 1, 2], [3, 4, 5]]], dtype=torch.float32),
    torch.tensor([[1, 2, 3]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32),
    torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=torch.float32)
]

if __name__ == "__main__":
    module = PtModule()
    for input_tensor in input_tensors:
        output_tensor = module(input_tensor)
        print(f"Input Tensor:\n{input_tensor}\nOutput Tensor:\n{output_tensor}\n")