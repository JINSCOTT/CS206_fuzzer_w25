import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        output = []
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                val = torch.sub(torch.pow(x[i, j], 2), 1)
                output.append(val)
        return torch.tensor(output).view(x.size(0), x.size(1))
input_tensors = [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]]), torch.tensor([[7.0, 8.0], [9.0, 10.0]]), torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]], [[[5.0, 6.0]], [[7.0, 8.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for (i, tensor) in enumerate(input_tensors):
        print(f'Input Tensor {torch.add(i, 1)}:\n{tensor}')
        output = model(tensor)
        print(f'Output after processing Tensor {torch.add(i, 1)}:\n{output}\n')