import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        return 2 * x + 3 - x / 4 + torch.power(x, 2)

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[5.0], [6.0]], [[7.0], [8.0]]]),  # 4D tensor
    torch.tensor([[9.0, 10.0], [11.0, 12.0]]),  # 2D tensor
    torch.tensor([[[13.0]]]),  # 3D tensor
    torch.tensor([[14.0, 15.0, 16.0], [17.0, 18.0, 19.0]])  # 2D tensor
]

if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input:\n{tensor}\nOutput:\n{output}\n")