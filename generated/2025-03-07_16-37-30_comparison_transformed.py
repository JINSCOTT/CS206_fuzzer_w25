import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        (a, b) = inputs
        addition = torch.add(a, b)
        subtraction = torch.sub(a, b)
        multiplication = torch.mul(a, b)
        division = torch.div(a, torch.add(b, 1e-05))
        greater_than = a > b
        less_than = a < b
        equal_to = a == b
        return (addition, subtraction, multiplication, division, greater_than, less_than, equal_to)
input_tensors = [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32), torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for (i, output) in enumerate(outputs):
        print(f'Output {torch.add(i, 1)}:\n{output}')