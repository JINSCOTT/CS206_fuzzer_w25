import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = []
        for tensor in x:
            summed = torch.sum(tensor)
            mean = torch.mean(tensor)
            multiplied = torch.mul(tensor, 2)
            result.append((summed, mean, multiplied))
        return result
input_tensors = [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]]]), torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for output in outputs:
        print(output)