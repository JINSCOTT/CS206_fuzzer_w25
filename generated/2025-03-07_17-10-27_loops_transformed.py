import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        sum_result = torch.sum(x, dim=1)
        mean_result = torch.mean(x, dim=0)
        prod_result = torch.prod(x, dim=2)
        output = []
        for i in range(x.size(0)):
            output.append(torch.add(x[i], sum_result[i]))
        return torch.add(torch.stack(output), mean_result)
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), torch.tensor([[[0.5], [1.5]], [[2.5], [3.5]], [[4.5], [5.5]]]), torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[[-1.0]], [[-2.0]]], [[[-3.0]], [[-4.0]]]])]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        output = module(tensor)
        print(output)