import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = {}
        (a, b, c, d, e) = inputs
        results['add'] = torch.add(a, b)
        results['subtract'] = torch.sub(a, b)
        results['multiply'] = torch.mul(a, b)
        results['divide'] = torch.div(a, torch.add(b, 1e-06))
        results['power'] = torch.pow(a, 2)
        results['equal'] = a == b
        results['greater_than'] = a > b
        results['less_than'] = a < b
        return results
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 4.0], [3.0, 2.0]]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]), torch.tensor([[5, 5], [5, 5]])]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)