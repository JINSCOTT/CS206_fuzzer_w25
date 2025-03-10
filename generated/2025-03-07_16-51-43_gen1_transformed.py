import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = torch.mul(torch.add(tensor, 2), 3)
            results.append(result)
        return results
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]), torch.tensor([[[10, 20, 30]], [[40, 50, 60]]]), torch.tensor([[1, 2], [3, 4], [5, 6]]), torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]])]
if __name__ == '__main__':
    module = PtModule()
    output = module(input_tensors)
    for (i, out) in enumerate(output):
        print(f'Output of tensor {i}: \n{out}')