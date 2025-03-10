import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.add(x, 2)
        x = torch.sub(x, 1)
        x = torch.mul(x, 3)
        x = torch.div(x, 2)
        return x
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[[9.0]]]), torch.tensor([[[[10.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for (i, tensor) in enumerate(input_tensors):
        output = model(tensor)
        print(f'Output for input tensor {torch.add(i, 1)}:\n{output}\n')