import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        add_result = torch.add(x, 2)
        sub_result = torch.sub(x, 2)
        mul_result = torch.mul(x, 2)
        div_result = torch.div(x, 2)
        gt_result = x > 0
        lt_result = x < 0
        eq_result = x == 0
        return (add_result, sub_result, mul_result, div_result, gt_result, lt_result, eq_result)
input_tensors = [torch.tensor([[1.0, -1.0], [3.0, -3.0]]), torch.tensor([[[2.0, 3.0], [4.0, 5.0]], [[-1.0, -2.0], [-3.0, -4.0]]]), torch.tensor([[[-1.0, 1.0], [0.0, 2.0]]]), torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        results = model(tensor)
        print(f'Input:\n{tensor}\nResults:\n{results}\n')