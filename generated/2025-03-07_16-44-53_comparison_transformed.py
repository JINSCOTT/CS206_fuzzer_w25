import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2, x3, x4, x5):
        addition = torch.add(x1, x2)
        subtraction = torch.sub(x3, x4)
        multiplication = torch.mul(x2, x5)
        division = torch.div(x4, torch.add(x5, 1e-08))
        comparison = x1 > x3
        return (addition, subtraction, multiplication, division, comparison)
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([1.0, 2.0, 3.0]), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(*input_tensors)
    for (i, output) in enumerate(outputs):
        print(f'Output {torch.add(i, 1)}:\n{output}\n')