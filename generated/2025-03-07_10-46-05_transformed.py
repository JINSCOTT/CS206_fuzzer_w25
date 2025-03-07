import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x1 = torch.add(x, 1)
        x2 = torch.sub(x, 1)
        x3 = torch.mul(x, 2)
        x4 = torch.div(x, 2)
        sum_result = 0
        for i in range(x.size(0)):
            sum_result += x[i].sum()
        return (x1, x2, x3, x4, sum_result)
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32), torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.float32), torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)]
if __name__ == '__main__':
    pt_module = PtModule()
    for input_tensor in input_tensors:
        output = pt_module(input_tensor)
        print(f'Output for input tensor {input_tensor}:\n{output}\n')