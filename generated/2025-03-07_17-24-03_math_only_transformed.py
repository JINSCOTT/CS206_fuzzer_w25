import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        return torch.sub(torch.mul(torch.add(x, 2), 3), torch.div(5, 2))
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]]), torch.tensor([[[11.0]], [[12.0]], [[13.0]]]), torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]), torch.tensor([[[[1.0, 2.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for (i, input_tensor) in enumerate(input_tensors):
        output = model(input_tensor)
        print(f'Output {torch.add(i, 1)}:\n{output}')