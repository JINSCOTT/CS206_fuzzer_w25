import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2):
        addition = x1 + x2
        subtraction = x1 - x2
        multiplication = x1 * x2
        division = x1 / (x2 + 1e-8)  # avoid division by zero
        comparison = x1 > x2

        return addition, subtraction, multiplication, division, comparison


# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),            # 2D tensor
    torch.tensor([[5, 6, 7], [8, 9, 10]]),     # 2D tensor
    torch.tensor([[[1, 2], [3, 4]]]),           # 3D tensor
    torch.tensor([[[[3, 4, 5], [6, 7, 8]]]]),  # 4D tensor
    torch.tensor([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors[0], input_tensors[1])
    print(output)