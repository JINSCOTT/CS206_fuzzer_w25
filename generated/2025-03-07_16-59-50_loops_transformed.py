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
        for i in range(x.size(0)):
            x[i] = torch.pow(x[i], 2)
        return x
input_tensors = [torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32), torch.tensor([[[[2, 3, 4], [5, 6, 7]], [[8, 9, 10], [11, 12, 13]]]], dtype=torch.float32), torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32), torch.tensor([[[1], [2], [3]], [[4], [5], [6]]], dtype=torch.float32), torch.tensor([[[[1]]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')