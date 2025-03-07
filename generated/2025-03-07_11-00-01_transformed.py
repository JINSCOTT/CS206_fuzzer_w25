import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        add_result = torch.add(x, 5)
        sub_result = torch.sub(x, 2)
        mul_result = torch.mul(x, 3)
        div_result = torch.div(x, 2)
        gt_result = x > 0
        lt_result = x < 10
        eq_result = x == 5
        return (add_result, sub_result, mul_result, div_result, gt_result, lt_result, eq_result)
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1], [2]], [[3], [4]]]), torch.tensor([[[[1, 2, 3], [4, 5, 6]]]]), torch.tensor([[0.5, 1.5], [2.5, 3.5]]), torch.tensor([[[1], [2], [3]], [[4], [5], [6]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')