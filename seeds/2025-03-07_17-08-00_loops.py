import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example of multiple math operations
        y = x + 2
        y = y * 3
        y = y - 1
        
        # Example of a loop
        for i in range(5):
            y = y + (i * 0.5)
        
        return y

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),  # 2D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]]]),  # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),  # another 2D tensor
    torch.tensor([[[[9]], [[10]]], [[[11]], [[12]]]])  # another 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input Tensor:\n{input_tensor}\nOutput Tensor:\n{output}\n")