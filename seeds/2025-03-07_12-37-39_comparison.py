import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        result_add = x + 2
        result_sub = x - 2
        result_mul = x * 2
        result_div = x / 2
        result_gt = x > 1
        result_lt = x < 1
        return result_add, result_sub, result_mul, result_div, result_gt, result_lt

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),  # 4D tensor
    torch.tensor([[5.0]]),  # 2D tensor 1x1
    torch.tensor([[2.0, 3.0], [4.0, 5.0]])  # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)