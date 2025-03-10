import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
        
    def forward(self, x):
        # Example of element-wise operations
        x = x + 2  # Addition
        x = x * 3  # Multiplication
        
        # Example of a loop
        for i in range(5):
            x = x - i  # Subtraction

        # Example of division
        x = x / 5  # Division
        
        return x

input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),
    torch.tensor([[[2, 2, 2, 2], [2, 2, 2, 2]], [[2, 2, 2, 2], [2, 2, 2, 2]]], dtype=torch.float32),
    torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.float32),
    torch.tensor([[[1, 0, 1], [0, 1, 0]], [[1, 1, 0], [0, 0, 1]]], dtype=torch.float32),
    torch.tensor([[[5, 5], [5, 5]], [[5, 5], [5, 5]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)