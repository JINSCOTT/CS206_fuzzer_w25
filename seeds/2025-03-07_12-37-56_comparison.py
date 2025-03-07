import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
    
    def forward(self, x):
        # Math operations
        addition = x + 2
        subtraction = x - 2
        multiplication = x * 2
        division = x / 2
        
        # Comparison operations
        greater_than = x > 1
        less_than = x < 1
        equal_to = x == 1
        
        return addition, subtraction, multiplication, division, greater_than, less_than, equal_to

# Input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
    torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32),
    torch.tensor([[1, 1], [2, 2]], dtype=torch.float32),
    torch.tensor([[1.0, 2.5], [3.0, 4.5]], dtype=torch.float32),
    torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        result = model(tensor)
        print(result)