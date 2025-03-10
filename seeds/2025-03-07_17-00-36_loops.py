import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Perform various math operations
        x = x + 2  # Addition
        x = x - 1  # Subtraction
        x = x * 3  # Multiplication
        x = x / 2  # Division
        
        # Example loop
        for i in range(5):
            x = x + i
        
        return x

# Input tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32),
    torch.tensor([[[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),
    torch.tensor([[[13, 14], [15, 16]], [[17, 18], [19, 20]]], dtype=torch.float32),
    torch.tensor([[[21]]], dtype=torch.float32),
    torch.tensor([[[22, 23, 24, 25]], [[26, 27, 28, 29]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input Tensor:\n{tensor}\nOutput Tensor:\n{output}\n")