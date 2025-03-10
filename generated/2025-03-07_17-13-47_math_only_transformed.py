import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.add(x, 1)
        x = torch.sub(x, 2)
        x = torch.mul(x, 3)
        x = torch.div(x, 4)
        return x
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]]), torch.tensor([[[11.0], [12.0]], [[13.0], [14.0]]]), torch.tensor([[15.0, 16.0], [17.0, 18.0], [19.0, 20.0]]), torch.tensor([[[[21.0]]], [[[22.0]]], [[[23.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for (i, input_tensor) in enumerate(input_tensors):
        output = model(input_tensor)
        print(f'Output for input tensor {torch.add(i, 1)}: {output}')