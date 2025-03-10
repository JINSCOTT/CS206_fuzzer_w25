import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example of various math operations
        x = x + 2  # Addition
        x = x * 3  # Multiplication
        x = x - 1  # Subtraction
        x = x / 4  # Division
        
        # Loop example: summing elements
        for i in range(x.size(0)):
            x[i] = x[i].sum()  # Summing each 3D tensor element-wise

        return x

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 2x2x2
    torch.tensor([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]]),                # 1x2x3
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]),      # 3x2x1
    torch.tensor([[[10.0, 20.0]], [[30.0, 40.0]], [[50.0, 60.0]], [[70.0, 80.0]]]),  # 4x1x2
    torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]]])   # 6x1x1
]

# Main section to check script
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")