import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            added = torch.add(input_tensor, 10)
            multiplied = torch.mul(added, 2)
            sum_value = torch.sum(multiplied)
            results.append(sum_value)
        return results
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)