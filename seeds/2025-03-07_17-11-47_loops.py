import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Using a loop to apply multiple operations
        for i in range(x.shape[0]):
            x[i] = x[i] + 2  # Addition
            x[i] = x[i] * 3  # Multiplication
            x[i] = x[i] - 5  # Subtraction
            x[i] = x[i] / 2  # Division
        return x

# Define input tensors with explicit values
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # Shape (1, 2, 2)
    torch.tensor([[[5.0], [6.0]], [[7.0], [8.0]]]),  # Shape (2, 2, 1)
    torch.tensor([[[9.0, 10.0, 11.0]]]),  # Shape (1, 1, 3)
    torch.tensor([[[12.0, 13.0]], [[14.0, 15.0]], [[16.0, 17.0]]]),  # Shape (3, 1, 2)
    torch.tensor([[[18.0]]])  # Shape (1, 1, 1)
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output_tensor = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output_tensor}\n")