import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = {}
        result['addition'] = torch.add(x, 2)
        result['subtraction'] = torch.sub(x, 2)
        result['multiplication'] = torch.mul(x, 2)
        result['division'] = torch.div(x, 2)
        result['greater_than'] = x > 1
        result['less_than'] = x < 1
        result['equal_to'] = x == 1
        return result
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([1.0, 2.0, 3.0, 4.0])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Input Tensor: {tensor.numpy()}')
        print(f'Output: {output}')