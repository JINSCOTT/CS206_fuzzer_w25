import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example normal math operations
        x = x + 1
        x = x * 2
        x = x - 3
        x = x / 5
        return x

input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),  # 2D tensor
    torch.tensor([1, 2, 3, 4], dtype=torch.float32),  # 1D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]]], dtype=torch.float32)  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output_tensor = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output_tensor}\n")