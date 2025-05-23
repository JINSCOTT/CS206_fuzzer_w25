import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        add_result = torch.add(x, 2)
        sub_result = torch.sub(x, 3)
        mul_result = torch.mul(x, 4)
        div_result = torch.div(x, 5)
        gt_result = x > 1
        lt_result = x < 2
        eq_result = x == 0
        return (add_result, sub_result, mul_result, div_result, gt_result, lt_result, eq_result)
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]), torch.tensor([[[[11.0]], [[12.0]]]]), torch.tensor([[[13.0, 14.0], [15.0, 16.0]], [[17.0, 18.0], [19.0, 20.0]]]), torch.tensor([[[[21.0, 22.0]], [[23.0, 24.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)