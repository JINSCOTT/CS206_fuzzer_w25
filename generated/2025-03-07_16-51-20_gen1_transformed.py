import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = torch.zeros_like(input_tensor)
        for i in range(input_tensor.shape[0]):
            result[i] = torch.pow(input_tensor[i], 2)
        return result.sum(dim=0)
input_tensors = [torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])]
if __name__ == '__main__':
    pt_module = PtModule()
    for input_tensor in input_tensors:
        output = pt_module(input_tensor)
        print(output)