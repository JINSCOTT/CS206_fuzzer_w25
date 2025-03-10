import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Applying various math operations
        x = x + 2        # Addition
        x = x - 1        # Subtraction
        x = x * 3        # Multiplication
        x = x / 4        # Division
        x = torch.pow(x, 2)  # Exponentiation
        
        # Looping through the tensor and applying a condition
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                if x[i, j] > 10:
                    x[i, j] = 10  # Clamping the value to 10

        return x

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D Tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),                     # 4D Tensor
    torch.tensor([[1, 2], [3, 4], [5, 6]]),                               # 2D Tensor
    torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]),                  # 3D Tensor
    torch.tensor([[[[0.5], [0.6]], [[0.7], [0.8]]]])                      # 4D Tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")