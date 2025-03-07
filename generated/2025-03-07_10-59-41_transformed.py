import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        addition = torch.add(x, 2)
        subtraction = torch.sub(x, 2)
        multiplication = torch.mul(x, 2)
        division = torch.div(x, 2)
        is_greater = x > 1
        is_less_equal = x <= 3
        return (addition, subtraction, multiplication, division, is_greater, is_less_equal)
input_tensors = [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], [[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]]), torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([1.0, 2.0, 3.0])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        result = model(tensor)
        print(result)