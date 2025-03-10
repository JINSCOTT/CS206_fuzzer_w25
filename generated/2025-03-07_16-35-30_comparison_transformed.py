import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        addition = torch.add(x, 2)
        subtraction = torch.sub(x, 3)
        multiplication = torch.mul(x, 4)
        division = torch.div(x, 5)
        greater_than = x > 1
        less_than = x < 2
        equal_to = x == 2
        return (addition, subtraction, multiplication, division, greater_than, less_than, equal_to)
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.float32), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[1], [2], [3], [4]], dtype=torch.float32), torch.tensor([[[[1], [2]], [[3], [4]]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for inp in input_tensors:
        result = model(inp)
        print(f'Input:\n{inp}\nResults:\n{result}\n')