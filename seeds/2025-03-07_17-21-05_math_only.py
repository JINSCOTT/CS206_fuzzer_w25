import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations: addition, subtraction, multiplication, and division
        a = x + 2
        b = x - 2
        c = x * 3
        d = x / 2
        return a, b, c, d

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),        # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),      # 3D tensor
    torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]),  # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),        # 2D tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])       # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        result = model(tensor)
        print("Input:\n", tensor)
        print("Output:\n", result)