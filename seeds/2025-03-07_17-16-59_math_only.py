import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Performing basic math operations
        x = x + 2  # Addition
        x = x - 1  # Subtraction
        x = x * 3  # Multiplication
        x = x / 2  # Division
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([1, 2, 3], dtype=torch.float32),
    torch.tensor([[1], [2], [3], [4]]], dtype=torch.float32),
    torch.tensor([[[[1]]]], dtype=torch.float32)
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    for i, tensor in enumerate(input_tensors):
        output = model(tensor)
        print(f"Output for input tensor {i}:")
        print(output)