import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        addition = torch.add(x, 5)
        subtraction = torch.sub(x, 2)
        multiplication = torch.mul(x, 3)
        division = torch.div(x, 4)
        greater_than = x > 3
        less_than = x < 5
        return (addition, subtraction, multiplication, division, greater_than, less_than)
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32), torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32), torch.tensor([[[[1, 2]], [[3, 4]]]], dtype=torch.float32), torch.tensor([[7, 8], [9, 10]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Input:\n{tensor}\nOutput:\n{output}\n')