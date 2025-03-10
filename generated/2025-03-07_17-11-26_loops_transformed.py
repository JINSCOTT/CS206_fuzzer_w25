import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.add(x, 2)
        x = torch.mul(x, 3)
        x = torch.sub(x, 1)
        x = torch.div(x, 4)
        for i in range(x.size(0)):
            x[i] = x[i].sum()
        return x
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]), torch.tensor([[[10.0, 20.0]], [[30.0, 40.0]], [[50.0, 60.0]], [[70.0, 80.0]]]), torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')