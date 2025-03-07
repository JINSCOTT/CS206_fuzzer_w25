import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Normal math operations
        return (x + 2) * 3 - 5 / 2

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),          # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), # 3D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),    # 2D tensor
    torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=torch.float32), # 3D tensor
    torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)       # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")