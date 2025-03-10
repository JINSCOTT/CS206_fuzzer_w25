import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Example operations
        output = input_tensor * 2  # Multiplication
        for i in range(3):  # Loop example
            output = output + i  # Addition in loop
        output = output - 5  # Subtraction
        output = output / 2  # Division
        return output

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32),  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output_tensor = model(tensor)
        print(f"Input Tensor:\n{tensor}\nOutput Tensor:\n{output_tensor}\n")