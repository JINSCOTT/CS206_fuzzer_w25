import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensors):
        results = []
        for tensor in input_tensors:
            results.append(torch.mul(tensor, 2))
        return results
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]), torch.tensor([[9, 10, 11, 12], [13, 14, 15, 16]]), torch.tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    for (i, o) in enumerate(output):
        print(f'Output {i}: {o}')