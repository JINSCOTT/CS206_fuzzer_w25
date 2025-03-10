import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            results.append(torch.sum(tensor))
        return results
input_tensors = [torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]), torch.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]), torch.tensor([[10, 20, 30], [40, 50, 60]]), torch.tensor([[1], [2], [3], [4]])]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)