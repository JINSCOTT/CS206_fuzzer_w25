import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.add(x, 2)
        x = torch.mul(x, 3)
        x = torch.sub(x, 5)
        x = torch.div(x, 2)
        for i in range(2):
            x = torch.add(x, i)
        return x
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[2.0, 3.0], [4.0, 5.0]]), torch.tensor([[[[1]]], [[[2]]], [[[3]]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Input: {tensor}\nOutput: {output}\n')