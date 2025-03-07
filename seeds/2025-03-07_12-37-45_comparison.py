import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Mathematical operations
        add_result = x + 2
        sub_result = x - 3
        mul_result = x * 4
        div_result = x / 5

        # Comparison operations
        gt_result = x > 1
        lt_result = x < 2
        eq_result = x == 0

        return add_result, sub_result, mul_result, div_result, gt_result, lt_result, eq_result

input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),  # 2D tensor
    torch.tensor([[[[11.0]], [[12.0]]]]),  # 4D tensor
    torch.tensor([[[13.0, 14.0], [15.0, 16.0]], [[17.0, 18.0], [19.0, 20.0]]]),  # 4D tensor
    torch.tensor([[[[21.0, 22.0]], [[23.0, 24.0]]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)