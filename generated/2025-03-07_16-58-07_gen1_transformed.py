import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = torch.add(tensor.mean(), tensor.std())
            results.append(result)
        return results
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])]
if __name__ == '__main__':
    module = PtModule()
    outputs = module(input_tensors)
    print(outputs)