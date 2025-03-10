import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.add(x, 2)
        x = torch.mul(x, 3)
        x = torch.sub(x, 1)
        x = torch.div(x, 4)
        return x
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32), torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.float32), torch.tensor([[[[10]], [[20]]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Input: {tensor}\nOutput: {output}\n')