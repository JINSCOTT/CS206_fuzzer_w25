import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations: addition, subtraction, multiplication, and division
        x1 = x + 2
        x2 = x - 1
        x3 = x * 3
        x4 = x / 2
        return x1, x2, x3, x4

input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), # 3D tensor
    torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=torch.float32), # 4D tensor
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), # 2D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32), # 2D tensor
    torch.tensor([[[1]], [[2]], [[3]], [[4]]], dtype=torch.float32) # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)