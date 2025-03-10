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
        x = x / 2  # Division
        return x

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),                            # 2D tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]),  # 3D tensor
    torch.tensor([[[[13.0, 14.0], [15.0, 16.0]]]]),                     # 4D tensor
    torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),                # 2D tensor of zeros
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])                # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for idx, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {idx}:\n{output}")