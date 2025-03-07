import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Applying multiple math operations
        x = x + 2  # Addition
        x = x * 3  # Multiplication
        x = x / 4  # Division
        x = x - 1  # Subtraction

        # Using a loop to apply additional operations
        for i in range(5):
            x = x + i  # Adding loop index

        return x

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[[5.0, 10.0, 15.0], [20.0, 25.0, 30.0]]]),  # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),  # 4D tensor
    torch.tensor([[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]])  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)