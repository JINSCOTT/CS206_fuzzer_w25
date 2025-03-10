import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            mean_value = torch.mean(tensor)
            std_value = torch.std(tensor)
            results.append((mean_value.item(), std_value.item()))
        return results
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]]), torch.tensor([[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]]]), torch.tensor([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]])]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)