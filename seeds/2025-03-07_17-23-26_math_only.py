import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example normal math operations
        x = x + 2      # addition
        x = x - 1      # subtraction
        x = x * 3      # multiplication
        x = x / 2      # division
        return x

input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),                 # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), # 4D tensor
    torch.tensor([[0.0, -1.0], [3.5, 2.5]]),                           # 2D tensor
    torch.tensor([[[2.0], [4.0]], [[6.0], [8.0]], [[10.0], [12.0]]])   # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input:\n{tensor}\nOutput:\n{output}\n")