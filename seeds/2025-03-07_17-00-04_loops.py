import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        output = x.clone()
        
        # Example 1: Addition
        output += 2
        
        # Example 2: Multiplication
        output *= 3
        
        # Example 3: Loop to apply a function
        for i in range(2):
            output = output - i
        
        # Example 4: Division
        output = output / 2
        
        return output

# Defined input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32),
    torch.tensor([[[10]], [[20]], [[30]], [[40]]], dtype=torch.float32),
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")