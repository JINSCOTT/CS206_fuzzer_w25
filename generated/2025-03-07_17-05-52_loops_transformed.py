import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.add(x, 2)
        x = torch.sub(x, 1)
        x = torch.mul(x, 3)
        x = torch.div(x, 4)
        x = torch.pow(x, 2)
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                if x[i, j] > 10:
                    x[i, j] = 10
        return x
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]), torch.tensor([[1, 2], [3, 4], [5, 6]]), torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]), torch.tensor([[[[0.5], [0.6]], [[0.7], [0.8]]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')