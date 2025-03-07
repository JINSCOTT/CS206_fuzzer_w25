import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = []
        for tensor in x:
            result.append(torch.add(tensor, 1))
        return result
input_tensors = [torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]), torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1], [2]], [[3], [4]]]), torch.tensor([10, 20, 30])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for i, output in enumerate(outputs):
        print(f'Output {i}: {output}')