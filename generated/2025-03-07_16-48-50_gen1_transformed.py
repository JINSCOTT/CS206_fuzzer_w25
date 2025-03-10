import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = torch.mul(input_tensor, 2)
        for i in range(result.size(0)):
            result[i] += i
        return result
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[5, 10, 15], [20, 25, 30], [35, 40, 45]]), torch.tensor([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Input Tensor:\n{tensor}\nOutput Tensor:\n{output}\n')