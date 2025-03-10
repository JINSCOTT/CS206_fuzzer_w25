import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Perform various mathematical operations and loops
        output = input_tensor * 2  # Multiplication
        for i in range(3):
            output = output + (input_tensor ** (i + 1))  # Addition with power
        output = output / 3  # Division
        return output

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),  # 4D tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[9.0, 10.0], [11.0, 12.0]]]]),  # 4D tensor
    torch.tensor([[[13.0, 14.0], [15.0, 16.0]]])   # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)