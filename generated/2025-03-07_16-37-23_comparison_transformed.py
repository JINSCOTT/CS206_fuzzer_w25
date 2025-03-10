import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        addition = torch.add(x, 10)
        subtraction = torch.sub(x, 5)
        multiplication = torch.mul(x, 2)
        division = torch.div(x, 2)
        comparison = x > 0
        return (addition, subtraction, multiplication, division, comparison)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[1.0, -1.0], [-2.0, 2.0]]), torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([[[1.0]], [[2.0]], [[3.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')