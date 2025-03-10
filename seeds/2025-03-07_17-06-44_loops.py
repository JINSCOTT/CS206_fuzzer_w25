import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
        
    def forward(self, x):
        # Example of multiple math operations
        x = x + 2
        x = x * 3
        x = x - 1
        
        # Example of loops
        for i in range(2):
            x = x / (i + 1)  # Simple operation in a loop
        
        return x

input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),               # 2D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]),  # 4D tensor
    torch.tensor([[5.5, 6.5], [7.5, 8.5]]),               # 2D tensor with float values
    torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])        # 2D tensor with more rows
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")