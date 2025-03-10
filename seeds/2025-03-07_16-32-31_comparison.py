import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2, x3, x4, x5):
        addition = x1 + x2
        subtraction = x3 - x4
        multiplication = x5 * x1
        division = x2 / (x3 + 1e-6)  # Adding a small value to avoid division by zero
        comparisons = (x4 > x5) & (x1 < x3)

        return addition, subtraction, multiplication, division, comparisons

# Defining input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[5.0], [6.0]], [[7.0], [8.0]]]),  # 3D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),  # 4D tensor
    torch.tensor([[10.0, 20.0], [30.0, 40.0]]),  # 2D tensor
    torch.tensor([[2.0, 3.0], [4.0, 5.0]])   # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(*input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output {i+1}: {output}")