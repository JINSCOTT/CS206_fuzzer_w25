import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = {}
        results['add'] = torch.add(inputs[0], inputs[1])
        results['subtract'] = torch.sub(inputs[2], inputs[3])
        results['multiply'] = torch.mul(inputs[0], inputs[4])
        results['divide'] = torch.div(inputs[1], torch.add(inputs[2], 1e-05))
        results['greater_than'] = inputs[0] > inputs[1]
        results['less_than'] = inputs[2] < inputs[3]
        results['equal'] = inputs[4] == inputs[0]
        return results
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[0, 1], [1, 0]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for (key, value) in outputs.items():
        print(f'{key}: {value}')