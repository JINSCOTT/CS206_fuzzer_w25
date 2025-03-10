import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.add(x, 1)
        x = torch.sub(x, 2)
        x = torch.mul(x, 3)
        x = torch.div(x, 4)
        return x
input_tensors = [torch.tensor([[[1, 2], [3, 4]]]), torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1], [2]], [[3], [4]]]), torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Input Tensor:\n{tensor}\nOutput Tensor:\n{output}\n')