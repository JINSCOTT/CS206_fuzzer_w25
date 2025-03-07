import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.add(x, 1)
        x = torch.mul(x, 2)
        x = torch.div(x, 3)
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                if x[i, j] > 0:
                    x[i, j] = torch.sub(x[i, j], 5)
        return x
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]), torch.tensor([[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]]), torch.tensor([[[[1.0, 0.0]], [[2.0, 3.0]], [[4.0, 5.0]]]]), torch.tensor([[[4.0, 3.0], [2.0, 1.0]], [[0.0, -1.0], [-2.0, -3.0]]])]
if __name__ == '__main__':
    pt_module = PtModule()
    for idx, tensor in enumerate(input_tensors):
        output = pt_module(tensor)
        print(f'Output for input tensor {idx}:\n{output}')