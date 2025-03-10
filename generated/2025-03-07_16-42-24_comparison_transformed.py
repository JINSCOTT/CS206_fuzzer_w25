import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = {}
        results['add'] = torch.add(inputs[0], inputs[1])
        results['subtract'] = torch.sub(inputs[0], inputs[1])
        results['multiply'] = torch.mul(inputs[0], inputs[1])
        results['divide'] = torch.div(inputs[0], torch.add(inputs[1], 1e-08))
        results['greater_than'] = inputs[0] > inputs[1]
        results['less_than'] = inputs[0] < inputs[1]
        results['equal'] = inputs[0] == inputs[1]
        return results
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[1.0, 1.0], [1.0, 1.0]]), torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]), torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    results = model(input_tensors[:2])
    print('Results:', results)