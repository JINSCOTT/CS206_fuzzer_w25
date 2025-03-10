import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        return torch.div(torch.mul(torch.add(x, 2), torch.sub(x, 1)), 3)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]), torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]), torch.tensor([1.0, 2.0, 3.0])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input: {input_tensor}\nOutput: {output}\n')