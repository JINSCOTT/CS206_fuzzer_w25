import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        results = {}
        results['add'] = torch.add(x, 5)
        results['subtract'] = torch.sub(x, 3)
        results['multiply'] = torch.mul(x, 2)
        results['divide'] = torch.div(x, 4)
        results['greater_than'] = x > 1
        results['less_than'] = x < 10
        return results
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]]), torch.tensor([[[0.0], [1.0]], [[2.0], [3.0]], [[4.0], [5.0]]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]), torch.tensor([[[9.0, 10.0, 11.0]], [[12.0, 13.0, 14.0]], [[15.0, 16.0, 17.0]]])]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        output = module(tensor)
        print(output)