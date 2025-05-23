import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.add(x, 2)
        x = torch.mul(x, 3)
        x = torch.div(x, 2)
        x = torch.sub(x, 1)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] = torch.sin(x[i, j])
        return x
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]), torch.tensor([[[[11.0], [12.0]], [[13.0], [14.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Input:\n{tensor}\nOutput:\n{output}\n')