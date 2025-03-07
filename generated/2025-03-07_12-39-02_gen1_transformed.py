import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = input_tensor.clone()
        for i in range(result.size(0)):
            result[i] = torch.mul(result[i], 2)
            result[i] = torch.add(result[i], 1)
        return result
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[9.0, 10.0], [11.0, 12.0]]]), torch.tensor([[[13.0, 14.0], [15.0, 16.0]]]), torch.tensor([[[17.0, 18.0], [19.0, 20.0]]])]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        output = module(tensor)
        print(output)