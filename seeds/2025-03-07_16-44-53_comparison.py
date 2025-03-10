import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2, x3, x4, x5):
        addition = x1 + x2
        subtraction = x3 - x4
        multiplication = x2 * x5
        division = x4 / (x5 + 1e-8)  # To prevent division by zero
        comparison = x1 > x3
        
        return addition, subtraction, multiplication, division, comparison

input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),          # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),    # 2D tensor
    torch.tensor([1.0, 2.0, 3.0]),              # 1D tensor
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(*input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output {i + 1}:\n{output}\n")
