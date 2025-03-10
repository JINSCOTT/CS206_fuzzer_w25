import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Perform normal math operations
        x = x + 5            # addition
        x = x * 2            # multiplication
        x = x - 3            # subtraction
        x = x / 4            # division
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32),    # 3D tensor
    torch.tensor([[1, 2, 3, 4]], dtype=torch.float32),               # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32),    # 4D tensor
    torch.tensor([[[[1]]]], dtype=torch.float32),                     # 4D tensor
    torch.tensor([[[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=torch.float32)  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for i, tensor in enumerate(input_tensors):
        output = model(tensor)
        print(f"Output for input tensor {i}:\n{output}")