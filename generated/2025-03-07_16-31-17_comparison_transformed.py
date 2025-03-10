import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2, x3, x4, x5):
        addition = torch.add(x1, x2)
        subtraction = torch.sub(x3, x4)
        multiplication = torch.mul(x1, x2)
        division = torch.div(x5, torch.add(x2, 1e-08))
        greater_than = x1 > x3
        less_than = x4 < x5
        equal_to = x1 == x2
        return (addition, subtraction, multiplication, division, greater_than, less_than, equal_to)
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]), torch.tensor([[[1, 2], [3, 4]]]), torch.tensor([[2, 3], [4, 5]]), torch.tensor([[[10], [20], [30]]])]
if __name__ == '__main__':
    model = PtModule()
    results = model(*input_tensors)
    for res in results:
        print(res)