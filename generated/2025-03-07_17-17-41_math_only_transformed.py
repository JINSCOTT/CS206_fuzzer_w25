import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        return torch.sub(torch.add(torch.mul(x, 2), 5), torch.div(3, 2))
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]), torch.tensor([1.0, 2.0, 3.0]), torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for (i, input_tensor) in enumerate(input_tensors):
        output = model(input_tensor)
        print(f'Output for input tensor {i}:\n{output}')