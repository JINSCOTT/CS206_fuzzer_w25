import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        results = {}
        results['addition'] = torch.add(x, 2)
        results['multiplication'] = torch.mul(x, 3)
        results['subtraction'] = torch.sub(x, 1)
        results['division'] = torch.div(x, 4)
        results['greater_than'] = x > 1
        results['less_than'] = x < 5
        return results
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([[[1], [2]], [[3], [4]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Input Tensor:\n{tensor}\nOutput:\n{output}\n')