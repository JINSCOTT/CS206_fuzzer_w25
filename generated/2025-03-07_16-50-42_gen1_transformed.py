import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = torch.mul(torch.add(tensor, 2), 3)
            results.append(result)
        return results
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[0.0, -1.0], [-2.0, -3.0]], [[-4.0, -5.0], [-6.0, -7.0]]]), torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]), torch.tensor([[[10.0]], [[20.0]], [[30.0]], [[40.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)