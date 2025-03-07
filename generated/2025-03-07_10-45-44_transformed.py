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
        for i in range(3):
            x = torch.add(x, i)
        return x
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[9.0, 10.0], [11.0, 12.0]]]]), torch.tensor([[13.0, 14.0], [15.0, 16.0]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)