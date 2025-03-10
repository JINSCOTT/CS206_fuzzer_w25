import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        y = torch.add(x, 2)
        y = torch.mul(y, 3)
        for i in range(x.size(0)):
            y[i] = torch.div(y[i], torch.add(i, 1))
        y = torch.clamp(y, min=0.0)
        return y
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, -1.0], [2.0, 2.0]]]), torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[[0.0, 0.1, 0.2]], [[0.3, 0.4, 0.5]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')