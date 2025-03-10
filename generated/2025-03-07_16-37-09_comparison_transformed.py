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
        power = torch.pow(x, 2)
        comparison = x > 1
        return (addition, subtraction, multiplication, division, power, comparison)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        result = model(input_tensor)
        print('Input Tensor:\n', input_tensor)
        print('Results:\n', result)