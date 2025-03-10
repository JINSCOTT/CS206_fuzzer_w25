import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example mathematical operations
        x = x + 2  # Addition
        x = x * 3  # Multiplication
        x = x - 1  # Subtraction
        x = x / 4  # Division

        # Implementing a simple loop
        for i in range(2):
            x = x + i  # Looping addition
        
        return x

# Defining 5 input tensors with explicit values
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32),
    torch.tensor([[1], [2], [3], [4]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32),
    torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32)
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {i + 1}:\n{output}")