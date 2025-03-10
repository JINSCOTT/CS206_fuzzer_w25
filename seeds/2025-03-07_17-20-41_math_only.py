import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example of ordinary math operations
        x = x + 5          # Addition
        x = x - 2          # Subtraction
        x = x * 3          # Multiplication
        x = x / 2          # Division
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),           # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),          # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),       # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),            # Another 2D tensor
    torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]])         # Another 3D tensor
]

# Main section to check if the module and tensors are runnable
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output_tensor = model(input_tensor)
        print("Input Tensor:\n", input_tensor)
        print("Output Tensor:\n", output_tensor)