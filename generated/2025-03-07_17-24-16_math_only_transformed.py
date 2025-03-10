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
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.float32), torch.tensor([1, 2, 3], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32), torch.tensor([[[[1]]], [[[2]]], [[[3]]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')