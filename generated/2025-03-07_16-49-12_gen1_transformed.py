import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = torch.mul(input_tensor, 2)
        for i in range(result.shape[0]):
            result[i] = torch.add(result[i], 1)
        return result
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[1, 2], [3, 4], [5, 6]]), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]]), torch.tensor([[[[1, 2], [3, 4]]]])]
if __name__ == '__main__':
    pt_module = PtModule()
    for tensor in input_tensors:
        output = pt_module(tensor)
        print('Input Tensor:\n', tensor)
        print('Output Tensor:\n', output)