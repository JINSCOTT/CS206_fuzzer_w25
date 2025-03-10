import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Perform some math operations and a for loop
        output_tensor = input_tensor * 2  # Example operation
        for i in range(1, 4):
            output_tensor += input_tensor / i  # Another operation
        return output_tensor

# Define 5 input tensors of 3 to 4 dimensions with explicit values
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[9, 10, 11], [12, 13, 14]], [[15, 16, 17], [18, 19, 20]]], dtype=torch.float32),
    torch.tensor([[[21, 22], [23, 24], [25, 26]]], dtype=torch.float32),
    torch.tensor([[[27, 28, 29, 30], [31, 32, 33, 34]]], dtype=torch.float32),
    torch.tensor([[[35], [36]], [[37], [38]], [[39], [40]]], dtype=torch.float32)
]

if __name__ == "__main__":
    pt_module = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = pt_module(input_tensor)
        print(f"Output for input tensor {i + 1}:\n{output}\n")