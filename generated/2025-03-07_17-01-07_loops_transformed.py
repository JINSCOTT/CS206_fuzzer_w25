import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = torch.zeros_like(x)
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                result[i, j] = torch.add(torch.mul(x[i, j], 2), 5)
        return result
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Input:\n{tensor}\nOutput:\n{output}\n')