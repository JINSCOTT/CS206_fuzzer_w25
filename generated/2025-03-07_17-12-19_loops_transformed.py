import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            result = torch.mul(input_tensor, 2)
            result = torch.add(result, 3)
            result = torch.sub(result, 1)
            result = torch.div(result, 2)
            results.append(result)
        return results
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]), torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]), torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]), torch.tensor([[[[11.0], [12.0]], [[13.0], [14.0]], [[15.0], [16.0]]]])]
if __name__ == '__main__':
    module = PtModule()
    outputs = module(input_tensors)
    for (i, output) in enumerate(outputs):
        print(f'Output for input tensor {i}:')
        print(output)