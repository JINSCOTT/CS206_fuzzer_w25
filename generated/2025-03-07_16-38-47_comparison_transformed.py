import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        addition = torch.add(x, 2)
        subtraction = torch.sub(x, 2)
        multiplication = torch.mul(x, 2)
        division = torch.div(x, 2)
        comparison = x > 1
        return (addition, subtraction, multiplication, division, comparison)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[9.0, 10.0], [11.0, 12.0]]]]), torch.tensor([[13.0, 14.0], [15.0, 16.0]]), torch.tensor([[[[17.0, 18.0]], [[19.0, 20.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutputs:\n{output}\n')