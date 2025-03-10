import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        return torch.sub(torch.add(torch.mul(x, 2), 3), torch.div(5, 2))
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]), torch.tensor([[[[1.0]], [[2.0]]]]), torch.tensor([0.0, 1.0, 2.0, 3.0]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')