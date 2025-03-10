import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example math operations
        y = x + 2  # Addition
        y = y * 3  # Multiplication
        y = y - 5  # Subtraction
        y = y / 2  # Division
        return y

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),      # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),     # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[5.0], [6.0], [7.0]]),           # 2D tensor
    torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])   # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {i+1}: {output}")