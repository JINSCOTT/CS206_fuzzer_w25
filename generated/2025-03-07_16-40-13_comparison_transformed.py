import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2):
        add_result = torch.add(x1, x2)
        sub_result = torch.sub(x1, x2)
        mul_result = torch.mul(x1, x2)
        div_result = torch.div(x1, torch.add(x2, 1e-08))
        gt_result = x1 > x2
        return (add_result, sub_result, mul_result, div_result, gt_result)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[[9.0]]])]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        print(tensor)
    result = module(input_tensors[0], input_tensors[3])
    print('Results:', result)