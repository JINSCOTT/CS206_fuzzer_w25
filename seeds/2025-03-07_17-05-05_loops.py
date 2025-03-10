import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations with loops and math operators
        output = []
        for i in range(x.size(0)):  # Loop over the first dimension
            temp = x[i] * 2  # Multiply by 2
            temp = temp + 1  # Add 1
            output.append(temp.sum(dim=0))  # Sum over the last dimension
        return torch.stack(output)

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]),  # 3D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]]]),  # 4D tensor
    torch.tensor([[[[1, 0], [0, 1]], [[1, 0], [0, 1]]], [[[0, 1], [1, 0]], [[0, 1], [1, 0]]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")