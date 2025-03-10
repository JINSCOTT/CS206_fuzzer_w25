import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        y = x + 2        # addition
        y = y * 3        # multiplication
        for i in range(x.size(0)):  # loop over the first dimension
            y[i] = y[i] / (i + 1)  # division
        
        y = torch.clamp(y, min=0.0)  # clamp to ensure non-negative outputs
        return y

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D
    torch.tensor([[[1.0, -1.0], [2.0, 2.0]]]),  # 3D
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),  # 4D
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),  # 2D
    torch.tensor([[[0.0, 0.1, 0.2]], [[0.3, 0.4, 0.5]]])  # 3D
]

# Main section
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")