import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = []
        for i in range(x.size(0)):
            temp = torch.mul(x[i], 2)
            temp = torch.add(temp, 3)
            result.append(temp)
        result = torch.stack(result)
        return result
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[9, 10], [11, 12]]], dtype=torch.float32), torch.tensor([[[13], [14]], [[15], [16]], [[17], [18]]], dtype=torch.float32), torch.tensor([[[19, 20, 21], [22, 23, 24]]], dtype=torch.float32), torch.tensor([[[25], [26], [27]], [[28], [29], [30]]], dtype=torch.float32)]
if __name__ == '__main__':
    module = PtModule()
    for input_tensor in input_tensors:
        output = module(input_tensor)
        print(output)