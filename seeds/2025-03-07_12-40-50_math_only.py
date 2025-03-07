import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Perform simple math operations
        x = x + 2      # Addition
        x = x - 1      # Subtraction
        x = x * 3      # Multiplication
        x = x / 4      # Division
        return x

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),        # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),         # 4D tensor
    torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),   # 2D tensor
    torch.tensor([[[[5.0]]]])                                  # 4D tensor
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)