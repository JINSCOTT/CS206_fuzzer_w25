import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Normal math operations
        result = (x + 2) * 3 - 5 / 2
        return result

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),  # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),  # 4D tensor
    torch.tensor([1.0, 2.0, 3.0]),  # 1D tensor
    torch.tensor([[[[1.0]]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for inp in input_tensors:
        output = model(inp)
        print(f"Input:\n{inp}\nOutput:\n{output}\n")