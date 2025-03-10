import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x1 = torch.add(x, 2)
        x2 = torch.sub(x, 3)
        x3 = torch.mul(x, 4)
        x4 = torch.div(x, 5)
        return (x1, x2, x3, x4)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]), torch.tensor([1.0, 2.0, 3.0]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Input Tensor:\n{tensor}\nOutput:\n{output}\n')