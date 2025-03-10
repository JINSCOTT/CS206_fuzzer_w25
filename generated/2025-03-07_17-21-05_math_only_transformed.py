import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        a = torch.add(x, 2)
        b = torch.sub(x, 2)
        c = torch.mul(x, 3)
        d = torch.div(x, 2)
        return (a, b, c, d)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        result = model(tensor)
        print('Input:\n', tensor)
        print('Output:\n', result)