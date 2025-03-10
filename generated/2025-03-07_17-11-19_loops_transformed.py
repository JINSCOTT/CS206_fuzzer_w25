import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x += 1
        x *= 2
        for i in range(3):
            x -= i
        x /= 2
        return x
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]]), torch.tensor([[[[15.0, 16.0], [17.0, 18.0]], [[19.0, 20.0], [21.0, 22.0]]]]), torch.tensor([[[23.0], [24.0]], [[25.0], [26.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)