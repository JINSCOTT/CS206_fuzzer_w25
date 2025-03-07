import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations:
        # Addition
        x1 = x + 1
        
        # Subtraction
        x2 = x - 1
        
        # Multiplication
        x3 = x * 2
        
        # Division
        x4 = x / 2
        
        # For loop to sum all elements of the tensor
        sum_result = 0
        for i in range(x.size(0)):
            sum_result += x[i].sum()
        
        return x1, x2, x3, x4, sum_result

# Example input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),   # 2D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.float32),  # Another 2D tensor
    torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)  # 1D tensor
]

if __name__ == "__main__":
    pt_module = PtModule()
    for input_tensor in input_tensors:
        output = pt_module(input_tensor)
        print(f"Output for input tensor {input_tensor}:\n{output}\n")