import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        for i in range(x.size(0)):
            x[i] = torch.add(torch.mul(x[i], 2), 1)
        return x
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])]

def main():
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)
if __name__ == '__main__':
    main()