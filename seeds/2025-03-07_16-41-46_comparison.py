import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example of various mathematical operations
        addition = x + 5
        subtraction = x - 3
        multiplication = x * 2
        division = x / 2
        power = x ** 2
        
        # Example of comparison operations
        greater_than = x > 2
        less_than = x < 4
        equal_to = x == 3
        
        return addition, subtraction, multiplication, division, power, greater_than, less_than, equal_to

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([1.0, 2.0, 3.0, 4.0]),  # 1D tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),  # 2D tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])  # 3D tensor
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        results = model(input_tensor)
        print(f"Input:\n{input_tensor}\nResults:\n{results}\n")