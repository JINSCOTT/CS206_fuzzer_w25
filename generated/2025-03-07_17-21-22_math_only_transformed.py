import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        y = torch.add(x, 2)
        y = torch.mul(y, 3)
        y = torch.sub(y, 5)
        y = torch.div(y, 4)
        return y
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[[1], [2]], [[3], [4]]]]), torch.tensor([[5, 6, 7], [8, 9, 10]]), torch.tensor([1.0, 2.0, 3.0])]
if __name__ == '__main__':
    model = PtModule()
    for (i, input_tensor) in enumerate(input_tensors):
        output = model(input_tensor)
        print(f'Output for input tensor {i}:\n{output}')