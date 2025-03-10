import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = input_tensor * 2  # Simple math operation
        for i in range(10):  # For loop
            result = result + i
        return result

# Input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),         # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[10, 20], [30, 40], [50, 60]], dtype=torch.float32), # 2D tensor
    torch.tensor([[[[0.1], [0.2]], [[0.3], [0.4]]]], dtype=torch.float32)  # 4D tensor
]

if __name__ == "__main__":
    pt_module = PtModule()
    for tensor in input_tensors:
        output = pt_module(tensor)
        print(output)