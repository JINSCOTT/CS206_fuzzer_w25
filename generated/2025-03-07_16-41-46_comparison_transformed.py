import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        addition = torch.add(x, 5)
        subtraction = torch.sub(x, 3)
        multiplication = torch.mul(x, 2)
        division = torch.div(x, 2)
        power = torch.pow(x, 2)
        greater_than = x > 2
        less_than = x < 4
        equal_to = x == 3
        return (addition, subtraction, multiplication, division, power, greater_than, less_than, equal_to)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        results = model(input_tensor)
        print(f'Input:\n{input_tensor}\nResults:\n{results}\n')