import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        return (torch.add(x, 2), torch.sub(x, 1), torch.mul(x, 3), torch.div(x, 4))
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1], [2]], [[3], [4]]]]), torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]), torch.tensor([1, 2, 3, 4, 5])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input: {input_tensor}\nOutput: {output}\n')