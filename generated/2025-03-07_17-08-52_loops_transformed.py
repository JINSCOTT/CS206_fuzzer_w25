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
        for i in range(x.size(0)):
            addition[i] = addition[i].sum()
            subtraction[i] = subtraction[i].sum()
        return (addition, subtraction, multiplication, division)
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]]), torch.tensor([[15.0, 16.0], [17.0, 18.0], [19.0, 20.0]]), torch.tensor([[[21.0]], [[22.0]], [[23.0]]]), torch.tensor([[[24.0, 25.0, 26.0, 27.0]]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors[0])
    print(outputs)