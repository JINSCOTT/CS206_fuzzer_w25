import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.add(x, 1)
        x = torch.mul(x, 2)
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                x[i][j] = torch.sub(x[i][j], 3)
        return x
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[5, 10], [15, 20], [25, 30]]), torch.tensor([[[[1, 0, 2], [4, 2, 8]]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')