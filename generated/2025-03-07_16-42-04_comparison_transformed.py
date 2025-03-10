import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2):
        addition = torch.add(x1, x2)
        subtraction = torch.sub(x1, x2)
        multiplication = torch.mul(x1, x2)
        division = torch.div(x1, torch.add(x2, 1e-08))
        comparison = x1 > x2
        return (addition, subtraction, multiplication, division, comparison)
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6, 7], [8, 9, 10]]), torch.tensor([[[1, 2], [3, 4]]]), torch.tensor([[[[3, 4, 5], [6, 7, 8]]]]), torch.tensor([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors[0], input_tensors[1])
    print(output)