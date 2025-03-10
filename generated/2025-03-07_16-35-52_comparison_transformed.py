import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        addition = torch.add(x, 2)
        subtraction = torch.sub(x, 3)
        multiplication = torch.mul(x, 4)
        division = torch.div(x, 5)
        greater_than = x > 1
        less_than = x < 5
        equal_to = x == 2
        return (addition, subtraction, multiplication, division, greater_than, less_than, equal_to)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])]
if __name__ == '__main__':
    module = PtModule()
    for (i, tensor) in enumerate(input_tensors):
        output = module(tensor)
        print(f'Output for input tensor {torch.add(i, 1)}: {output}')