import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        addition = torch.add(input_tensor, 2)
        subtraction = torch.sub(input_tensor, 2)
        multiplication = torch.mul(input_tensor, 2)
        division = torch.div(input_tensor, 2)
        greater_than = input_tensor > 1
        less_than = input_tensor < 3
        equal_to = input_tensor == 2
        return (addition, subtraction, multiplication, division, greater_than, less_than, equal_to)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([1.0, 2.0, 3.0])]
if __name__ == '__main__':
    module = PtModule()
    for inp in input_tensors:
        results = module(inp)
        print(f'Input: {inp}\nResults: {results}\n')