import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2, x3, x4, x5):
        # Performing normal math operations
        add = x1 + x2
        sub = x3 - x4
        mul = x1 * x5
        div = x4 / (x5 + 1e-5)  # Adding a small value to avoid division by zero
        power = x2 ** 2
        return add, sub, mul, div, power

# Input tensors with explicitly defined values
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),                      # 2D tensor
    torch.tensor([[5, 6], [7, 8]], dtype=torch.float32),                      # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32)              # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    results = model(*input_tensors)
    for result in results:
        print(result)