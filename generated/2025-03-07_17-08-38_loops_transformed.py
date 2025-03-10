import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = input_tensor.clone()
        for i in range(result.size(0)):
            for j in range(result.size(1)):
                result[i][j] += 1
                result[i][j] *= 2
                result[i][j] -= 3
        return result
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]), torch.tensor([[[17, 18], [19, 20]], [[21, 22], [23, 24]]]), torch.tensor([[[25, 26], [27, 28]], [[29, 30], [31, 32]]]), torch.tensor([[[33, 34], [35, 36]], [[37, 38], [39, 40]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print('Output Tensor:', output)