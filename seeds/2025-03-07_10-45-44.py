import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example of using various math operators
        x = x + 1  # Addition
        x = x - 2  # Subtraction
        x = x * 3  # Multiplication
        x = x / 4  # Division

        # Example of a loop
        for i in range(3):
            x = x + i  # Incrementally add loop index

        return x

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),  # 4D tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[9.0, 10.0], [11.0, 12.0]]]]),  # 4D tensor
    torch.tensor([[13.0, 14.0], [15.0, 16.0]])  # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)