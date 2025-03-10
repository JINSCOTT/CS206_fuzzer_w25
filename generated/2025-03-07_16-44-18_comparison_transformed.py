import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        addition = torch.add(x, 5)
        subtraction = torch.sub(x, 3)
        multiplication = torch.mul(x, 2)
        division = torch.div(x, 4)
        greater_than = x > 2
        less_than = x < 10
        equal_to = x == 3
        return {'addition': addition, 'subtraction': subtraction, 'multiplication': multiplication, 'division': division, 'greater_than': greater_than, 'less_than': less_than, 'equal_to': equal_to}
input_tensors = [torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]), torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]]), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]])]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        result = module(tensor)
        print(result)