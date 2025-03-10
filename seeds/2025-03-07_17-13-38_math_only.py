import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
        
    def forward(self, x):
        return (x + 2) * 3 - 1 / 2

input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),  # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]]),  # 3D tensor
    torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]]),  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {i}:\n{output}")