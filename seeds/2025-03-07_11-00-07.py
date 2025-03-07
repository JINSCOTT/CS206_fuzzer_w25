import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example math operations
        addition = x + 5
        subtraction = x - 3
        multiplication = x * 2
        division = x / 4
        
        # Example comparison operations
        greater_than = x > 0
        less_than = x < 10
        
        return addition, subtraction, multiplication, division, greater_than, less_than

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),          # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]]]]),      # 4D tensor
    torch.tensor([[5, 6, 7], [8, 9, 10]]),    # 2D tensor
    torch.tensor([1, 2, 3, 4, 5])             # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input:\n{tensor}\nOutput:\n{output}\n")