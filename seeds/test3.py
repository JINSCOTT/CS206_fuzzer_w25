import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Using basic math operations
        result = x + 5  # Addition
        result = result * 2  # Multiplication

        # Loop through the tensor dimensions
        for i in range(result.size(0)):
            for j in range(result.size(1)):
                result[i, j] = result[i, j] - 1  # Subtraction in a loop

        return result

# Defining input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),  # 4D tensor
    torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),  # 3D tensor of floats
    torch.tensor([[[[1], [2]], [[3], [4]]]]),  # 4D tensor with single value in last dimension
    torch.tensor([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]])  # 4D tensor of ones
]

if __name__ == "__main__":
    pt_module = PtModule()
    for input_tensor in input_tensors:
        output = pt_module(input_tensor)
        print(output)