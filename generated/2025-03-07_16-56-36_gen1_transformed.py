import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = torch.sum(tensor)
            results.append(result)
        return results
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([1, 2, 3, 4, 5]), torch.tensor([[[[1, 2], [3, 4]]]]), torch.tensor([[10, 20, 30], [40, 50, 60]])]
if __name__ == '__main__':
    model = PtModule()
    results = model(input_tensors)
    for (i, res) in enumerate(results):
        print(f'Result for input tensor {i}: {res.item()}')