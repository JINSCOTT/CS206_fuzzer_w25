import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = torch.sub(torch.mul(torch.add(x, 2), 3), torch.div(5, 2))
        return result
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]), torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]), torch.tensor([1.0, 2.0, 3.0]), torch.tensor([[[[1.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for inp in input_tensors:
        output = model(inp)
        print(f'Input:\n{inp}\nOutput:\n{output}\n')