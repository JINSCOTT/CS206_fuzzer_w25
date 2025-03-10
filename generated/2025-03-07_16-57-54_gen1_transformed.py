import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = []
        for i in range(x.shape[0]):
            processed = torch.add(torch.mul(x[i], 2), 1)
            result.append(processed)
        return torch.stack(result)
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]), torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]], [[[5.0, 6.0]]]])]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        output = module(tensor)
        print(output)