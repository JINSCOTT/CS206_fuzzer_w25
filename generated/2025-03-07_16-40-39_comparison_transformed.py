import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = {}
        (a, b, c, d, e) = inputs
        results['addition'] = torch.add(a, b)
        results['subtraction'] = torch.sub(a, c)
        results['multiplication'] = torch.mul(b, d)
        results['division'] = torch.div(b, torch.add(d, 1e-06))
        results['greater_than'] = a > e
        results['less_than'] = b < d
        results['equal'] = c == e
        return results
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32), torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32), torch.tensor([[9.0, 10.0]], dtype=torch.float32), torch.tensor([[[11.0, 12.0]]], dtype=torch.float32), torch.tensor([[2.0, 3.0]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for (key, value) in outputs.items():
        print(f'{key}: {value}')