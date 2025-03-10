import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Normal math operations
        y = x + 2  # addition
        z = y * 3  # multiplication
        t = z - 5  # subtraction
        u = t / 2  # division
        return u

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[5.0], [6.0], [7.0]]),  # 2D tensor
    torch.tensor([1.0, 2.0, 3.0, 4.0])  # 1D tensor
]

# Main section to check runnable
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input: {input_tensor}, Output: {output}")