import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x1 = torch.add(x, 2)
        x2 = torch.sub(x, 1)
        x3 = torch.mul(x, 3)
        x4 = torch.div(x, 2)
        return (x1, x2, x3, x4)
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[9.0, 10.0], [11.0, 12.0]]), torch.tensor([[[13.0], [14.0]], [[15.0], [16.0]]]), torch.tensor([[[[17.0], [18.0]], [[19.0], [20.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print('Input Tensor:', tensor)
        print('Output:', output)