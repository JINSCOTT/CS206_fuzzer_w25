import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example of basic mathematical operations 
        x = x + 1  # Addition
        x = x - 1  # Subtraction
        x = x * 2  # Multiplication
        x = x / 2  # Division
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),                 # 2D Tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D Tensor
    torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]),  # 4D Tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),    # 2D Tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])  # 4D Tensor
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output_tensor = model(input_tensor)
        print(f"Output for input tensor {i}: {output_tensor}")