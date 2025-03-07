import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        return (x + 2) * 3 - 5 / 2

input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),        # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]],        # 3D tensor
                   [[5.0, 6.0], [7.0, 8.0]]]),
    torch.tensor([[[[1.0], [2.0]],                  # 4D tensor
                   [[[3.0], [4.0]]]]]),
    torch.tensor([[1, 2], [3, 4], [5, 6]]),        # 2D tensor with integers
    torch.tensor([[[[2]]], [[[3]]]], dtype=torch.float)  # 4D tensor with float
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input: {input_tensor}\nOutput: {output}\n")