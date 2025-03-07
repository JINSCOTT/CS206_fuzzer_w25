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
        greater_than = x > 0
        less_than = x < 10
        return (addition, subtraction, multiplication, division, greater_than, less_than)
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1], [2]], [[3], [4]]]), torch.tensor([[[[1, 2], [3, 4]]]]), torch.tensor([[5, 6, 7], [8, 9, 10]]), torch.tensor([1, 2, 3, 4, 5])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Input:\n{tensor}\nOutput:\n{output}\n')