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
        less_than = x < 2
        greater_than = x > 2
        equal_to = x == 2
        return {'addition': addition, 'subtraction': subtraction, 'multiplication': multiplication, 'division': division, 'less_than': less_than, 'greater_than': greater_than, 'equal_to': equal_to}
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[1, 2], [3, 4], [5, 6]]), torch.tensor([1.0, 2.0, 3.0]), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]])]
if __name__ == '__main__':
    model = PtModule()
    for (i, tensor) in enumerate(input_tensors):
        output = model(tensor)
        print(f'Output for input tensor {torch.add(i, 1)}:\n{output}\n')