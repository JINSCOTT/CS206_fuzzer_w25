import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        y = torch.add(x, 2)
        y = torch.mul(y, 3)
        y = torch.sub(y, 1)
        for i in range(5):
            y = torch.add(y, torch.mul(i, 0.5))
        return y
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[[1], [2]], [[3], [4]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[[[9]], [[10]]], [[[11]], [[12]]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input Tensor:\n{input_tensor}\nOutput Tensor:\n{output}\n')