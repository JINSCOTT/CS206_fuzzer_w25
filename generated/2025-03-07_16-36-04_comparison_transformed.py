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
        gt_result = x > 1
        lt_result = x < 1
        eq_result = x == 1
        return (add_result, sub_result, mul_result, div_result, gt_result, lt_result, eq_result)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]), torch.tensor([[1.0, -1.0, 3.0], [2.0, 0.0, -2.0]]), torch.tensor([[[1.0, -1.0], [2.0, 3.0]], [[-2.0, -3.0], [4.0, 0.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        results = model(input_tensor)
        print(f'Input:\n{input_tensor}\nResults:\n{results}\n')