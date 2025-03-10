import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
    
    def forward(self, x):
        # Mathematical operations
        addition = x + 2
        subtraction = x - 3
        multiplication = x * 4
        division = x / 5
        
        # Comparison operations
        greater_than = x > 1
        less_than = x < 5
        equal_to = x == 2
        
        return addition, subtraction, multiplication, division, greater_than, less_than, equal_to

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),  # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),  # 2D tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])   # 3D tensor
]

if __name__ == "__main__":
    module = PtModule()
    for i, tensor in enumerate(input_tensors):
        output = module(tensor)
        print(f"Output for input tensor {i + 1}: {output}")