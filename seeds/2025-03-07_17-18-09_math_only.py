import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Normal math operations
        x = x + 2  # Addition
        x = x - 1  # Subtraction
        x = x * 3  # Multiplication
        x = x / 2  # Division
        return x

# Input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),            # 3D tensor
    torch.tensor([[[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]]]),  # 4D tensor
    torch.tensor([[11.0, 12.0, 13.0]]),                    # 2D tensor
    torch.tensor([4.0]),                                   # 1D tensor
    torch.tensor([[14.0], [15.0], [16.0]])                 # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output_tensor = model(input_tensor)
        print(output_tensor)