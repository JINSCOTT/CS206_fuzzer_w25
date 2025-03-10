import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.add(x, 2)
        x = torch.sub(x, 3)
        x = torch.mul(x, 4)
        x = torch.div(x, 2)
        return x
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1], [2]], [[3], [4]]]]), torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]]), torch.tensor([[[[1, 2]], [[3, 4]], [[5, 6]]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print('Input:\n', tensor)
        print('Output:\n', output)