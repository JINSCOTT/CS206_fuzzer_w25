import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2):
        results = {}
        results['addition'] = torch.add(x1, x2)
        results['subtraction'] = torch.sub(x1, x2)
        results['multiplication'] = torch.mul(x1, x2)
        results['division'] = torch.div(x1, torch.add(x2, 1e-10))
        results['greater_than'] = x1 > x2
        results['less_than'] = x1 < x2
        results['equal_to'] = x1 == x2
        return results
input_tensors = [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]]]), torch.tensor([[1.0], [0.0], [1.0]])]
if __name__ == '__main__':
    model = PtModule()
    x1 = input_tensors[0]
    x2 = input_tensors[1]
    output = model(x1, x2)
    print(output)