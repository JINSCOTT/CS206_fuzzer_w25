import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example mathematical operations
        x = x + 2  # Addition
        x = x - 1  # Subtraction
        x = x * 3  # Multiplication
        x = x / 2  # Division
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),                # 2D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),   # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),      # 4D tensor
    torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),     # 2D tensor
    torch.tensor([[[5.0, 6.0, 7.0]], [[8.0, 9.0, 10.0]]]) # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for i, tensor in enumerate(input_tensors):
        output = model(tensor)
        print(f"Output for input tensor {i}:\n{output}")