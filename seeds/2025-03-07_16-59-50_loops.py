import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example math operations
        x = x + 2          # Addition
        x = x * 3          # Multiplication
        x = x - 1          # Subtraction
        x = x / 4          # Division

        # Example loop operation
        for i in range(x.size(0)):
            x[i] = x[i] ** 2  # Squaring each element

        return x

# Input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32),  # 3D
    torch.tensor([[[[2, 3, 4], [5, 6, 7]], [[8, 9, 10], [11, 12, 13]]]], dtype=torch.float32),  # 4D
    torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32),  # 2D
    torch.tensor([[[1], [2], [3]], [[4], [5], [6]]], dtype=torch.float32),  # 3D
    torch.tensor([[[[1]]]], dtype=torch.float32)  # 4D
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")