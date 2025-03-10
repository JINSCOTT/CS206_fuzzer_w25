import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Addition
        addition = x + 5
        
        # Subtraction
        subtraction = x - 2
        
        # Multiplication
        multiplication = x * 3
        
        # Division
        division = x / 4
        
        # Comparisons
        greater_than = x > 3
        less_than = x < 5
        
        return addition, subtraction, multiplication, division, greater_than, less_than

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[[1, 2]], [[3, 4]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[7, 8], [9, 10]], dtype=torch.float32)  # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input:\n{tensor}\nOutput:\n{output}\n")