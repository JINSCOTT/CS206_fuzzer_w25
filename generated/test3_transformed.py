import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = torch.add(x, 5)
        result = torch.mul(result, 2)
        for i in range(result.size(0)):
            for j in range(result.size(1)):
                result[i, j] = torch.sub(result[i, j], 1)
        return result
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]), torch.tensor([[[[1], [2]], [[3], [4]]]]), torch.tensor([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]])]
if __name__ == '__main__':
    pt_module = PtModule()
    for input_tensor in input_tensors:
        output = pt_module(input_tensor)
        print(output)