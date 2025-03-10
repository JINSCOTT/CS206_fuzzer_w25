import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        return x * 2 + 3 - 1 / 2

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor
    torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]),  # 4D tensor
    torch.tensor([5.0]),  # 1D tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)