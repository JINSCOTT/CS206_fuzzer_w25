import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        addition = torch.add(x, 2)
        subtraction = torch.sub(x, 3)
        multiplication = torch.mul(x, 4)
        division = torch.div(x, 2)
        power = torch.pow(x, 2)
        greater_than = x > 1
        less_than = x < 5
        equal_to = x == 3
        return (addition, subtraction, multiplication, division, power, greater_than, less_than, equal_to)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[1.0, 4.0, 2.0], [3.0, 5.0, 7.0]]), torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]]]]]), torch.tensor([[10.0, 5.0], [3.0, 8.0], [2.0, 1.0]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)