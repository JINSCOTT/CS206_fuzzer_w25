import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Normal math operations using ordinary python math operators
        x = x + 2  # Addition
        x = x - 1  # Subtraction
        x = x * 3  # Multiplication
        x = x / 2  # Division
        return x

# Input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[5.0, 6.0]], [[7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor
    torch.tensor([[[9.0]]]),  # 3D tensor with one element
    torch.tensor([[10.0, 20.0], [30.0, 40.0]])  # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    
    # Testing the module with the input tensors
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {i}:\n{output}")