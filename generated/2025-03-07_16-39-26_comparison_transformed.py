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
        result['power'] = torch.pow(x, 2)
        result['greater_than'] = x > 1
        result['less_than'] = x < 1
        return result
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32), torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=torch.float32), torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        print(f'Input:\n{tensor}\nOutput:\n{model(tensor)}\n')