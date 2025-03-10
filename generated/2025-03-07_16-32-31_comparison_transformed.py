import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2, x3, x4, x5):
        addition = torch.add(x1, x2)
        subtraction = torch.sub(x3, x4)
        multiplication = torch.mul(x5, x1)
        division = torch.div(x2, torch.add(x3, 1e-06))
        comparisons = (x4 > x5) & (x1 < x3)
        return (addition, subtraction, multiplication, division, comparisons)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[5.0], [6.0]], [[7.0], [8.0]]]), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]), torch.tensor([[10.0, 20.0], [30.0, 40.0]]), torch.tensor([[2.0, 3.0], [4.0, 5.0]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(*input_tensors)
    for (i, output) in enumerate(outputs):
        print(f'Output {torch.add(i, 1)}: {output}')