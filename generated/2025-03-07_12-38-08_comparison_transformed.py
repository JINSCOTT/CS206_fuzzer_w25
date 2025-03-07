import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2):
        addition = torch.add(x1, x2)
        subtraction = torch.sub(x1, x2)
        multiplication = torch.mul(x1, x2)
        division = torch.div(x1, x2)
        greater_than = x1 > x2
        less_than = x1 < x2
        equal_to = x1 == x2
        return (addition, subtraction, multiplication, division, greater_than, less_than, equal_to)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors[0], input_tensors[3])
    for output in outputs:
        print(output)