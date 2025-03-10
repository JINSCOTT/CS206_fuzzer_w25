import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        addition = x + 2
        subtraction = x - 2
        multiplication = x * 2
        division = x / 2
        power = x ** 2
        comparison = x > 1
        
        return addition, subtraction, multiplication, division, power, comparison

# Input Tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),               # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6]]),                 # 2D tensor
    torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])   # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        result = model(input_tensor)
        print("Input Tensor:\n", input_tensor)
        print("Results:\n", result)