import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        addition = x + 10
        subtraction = x - 5
        multiplication = x * 2
        division = x / 2
        comparison = x > 0
        
        return addition, subtraction, multiplication, division, comparison

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[1.0, -1.0], [-2.0, 2.0]]),  # 2D tensor with negative values
    torch.tensor([1.0, 2.0, 3.0, 4.0]),  # 1D tensor
    torch.tensor([[[1.0]], [[2.0]], [[3.0]]])  # 3D tensor with single values
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")