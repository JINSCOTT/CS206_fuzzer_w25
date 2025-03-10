import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = torch.add(tensor, 1)
            results.append(result)
        return results
input_tensors = [torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]), torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]), torch.tensor([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]], [[10], [11], [12]]]), torch.tensor([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]])]
if __name__ == '__main__':
    module = PtModule()
    output = module(input_tensors)
    for (i, o) in enumerate(output):
        print(f'Output tensor {i}: \n{o}')