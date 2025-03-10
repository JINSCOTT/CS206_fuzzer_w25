import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        y = torch.add(x, 2)
        y = torch.mul(y, 3)
        y = torch.sub(y, 1)
        y = torch.div(y, 4)
        return y
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32), torch.tensor([[[[1, 2]], [[3, 4]]]], dtype=torch.float32), torch.tensor([[0, -1, -2], [3, 4, 5]], dtype=torch.float32), torch.tensor([[[0.1], [0.2]], [[0.3], [0.4]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')