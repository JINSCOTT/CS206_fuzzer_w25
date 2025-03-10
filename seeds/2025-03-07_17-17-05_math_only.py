import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
    
    def forward(self, x):
        return x + 2, x - 1, x * 3, x / 4

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]]]),  # 4D tensor
    torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),  # 2D tensor
    torch.tensor([1, 2, 3, 4, 5])  # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input: {input_tensor}\nOutput: {output}\n")
