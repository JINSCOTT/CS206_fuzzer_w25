import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x1 = torch.add(x, 2)
        x2 = torch.mul(x, 3)
        x3 = torch.sub(x, 1)
        x4 = torch.div(x, 2)
        result = torch.zeros_like(x)
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                result[i, j] = torch.sub(torch.add(x1[i, j], x2[i, j]), torch.mul(x3[i, j], x4[i, j]))
        return result
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]), torch.tensor([[[[5.0, 6.0]], [[7.0, 8.0]], [[9.0, 10.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input: {input_tensor}\nOutput: {output}\n')