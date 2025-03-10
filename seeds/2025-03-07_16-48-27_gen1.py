import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = input_tensor * 2  # Simple math operation
        for i in range(input_tensor.shape[0]):  # Loop over the first dimension
            result[i] += 1  # Another operation
        return result

# Input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # Shape: (2, 2, 2)
    torch.tensor([[1, 2, 3], [4, 5, 6]]),  # Shape: (2, 3)
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]]),  # Shape: (3, 2, 1)
    torch.tensor([[[1, 2, 3]], [[4, 5, 6]]]),  # Shape: (2, 1, 3)
    torch.tensor([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]])  # Shape: (2, 1, 2, 2)
]

if __name__ == "__main__":
    module = PtModule()
    for input_tensor in input_tensors:
        output = module(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")