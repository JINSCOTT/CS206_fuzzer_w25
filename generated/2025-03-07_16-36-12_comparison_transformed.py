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
        power = torch.pow(x, 2)
        greater_than = x > 2
        less_than = x < 10
        equal_to = x == 5
        return {'addition': addition, 'subtraction': subtraction, 'multiplication': multiplication, 'division': division, 'power': power, 'greater_than': greater_than, 'less_than': less_than, 'equal_to': equal_to}
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]]), torch.tensor([[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]), torch.tensor([[[15.0, 16.0, 17.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)