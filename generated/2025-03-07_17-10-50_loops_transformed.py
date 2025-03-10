import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = torch.add(torch.mul(input_tensor, 2), 3)
        for i in range(result.size(0)):
            for j in range(result.size(1)):
                result[i, j] = torch.div(result[i, j], torch.add(i, 1))
        return result
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]), torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]), torch.tensor([[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)