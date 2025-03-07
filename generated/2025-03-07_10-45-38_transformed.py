import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = torch.mul(input_tensor, 2)
        result = torch.add(result, 5)
        for i in range(result.size(0)):
            for j in range(result.size(1)):
                result[i, j] = torch.sub(result[i, j], torch.add(i, j))
        return result
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[10, 20], [30, 40]]), torch.tensor([[[[9], [10]], [[11], [12]]], [[[13], [14]], [[15], [16]]]])]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        output = module(tensor)
        print(f'Input Tensor:\n{tensor}\nOutput Tensor:\n{output}\n')