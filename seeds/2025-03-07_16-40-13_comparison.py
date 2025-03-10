import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2):
        # Addition
        add_result = x1 + x2
        # Subtraction
        sub_result = x1 - x2
        # Multiplication
        mul_result = x1 * x2
        # Division
        div_result = x1 / (x2 + 1e-8)  # prevent division by zero
        # Comparison (greater than)
        gt_result = x1 > x2

        return add_result, sub_result, mul_result, div_result, gt_result

input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]),  # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),  # another 2D tensor
    torch.tensor([[[9.0]]])  # another 3D tensor
]

if __name__ == "__main__":
    module = PtModule()
    for tensor in input_tensors:
        print(tensor)
    result = module(input_tensors[0], input_tensors[3])
    print("Results:", result)