import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        add_result = torch.add(x, 5)
        sub_result = torch.sub(x, 3)
        mul_result = torch.mul(x, 2)
        div_result = torch.div(x, 2)
        gt_result = x > 0
        lt_result = x < 0
        return (add_result, sub_result, mul_result, div_result, gt_result, lt_result)
input_tensors = [torch.tensor([[1, -2], [3, -4]], dtype=torch.float32), torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32), torch.tensor([[-1, 0], [1, 2]], dtype=torch.float32), torch.tensor([[[[2]]]], dtype=torch.float32), torch.tensor([[[[3, 4], [5, 6]], [[7, 8], [9, 10]]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        result = model(tensor)
        print('Input Tensor:\n', tensor)
        print('Results:', result)