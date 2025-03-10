import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        sum_result = x.sum(dim=1)
        product_result = x.prod(dim=1)
        max_result = x.max(dim=1).values
        min_result = x.min(dim=1).values
        mean_result = x.mean(dim=1)
        for i in range(x.shape[0]):
            sum_result[i] += 1
        return (sum_result, product_result, max_result, min_result, mean_result)
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]), torch.tensor([[[0.5, 1.5]], [[2.5, 3.5]], [[4.5, 5.5]]]), torch.tensor([[[10], [20]], [[30], [40]], [[50], [60]], [[70], [80]]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Output for tensor {tensor}:\n{output}\n')