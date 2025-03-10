import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Applying multiple math operations
        y = x + 2         # Addition
        y = y * 3         # Multiplication
        y = y - 1         # Subtraction
        y = y / 4         # Division
        
        # Loop through dimensions
        batch_size, channels, height, width = y.shape
        for i in range(batch_size):
            for j in range(channels):
                # Example operation: square each element
                y[i, j] = y[i, j] ** 2
        
        return y

# Input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32),
    torch.tensor([[[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[9, 10], [11, 12]]], dtype=torch.float32),
    torch.tensor([[[13, 14], [15, 16]]], dtype=torch.float32),
    torch.tensor([[[17, 18], [19, 20]]], dtype=torch.float32),
]

# Main
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)