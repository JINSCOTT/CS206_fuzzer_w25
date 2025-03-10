import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            if tensor.dim() == 3:
                results.append(torch.add(tensor, 1))
            elif tensor.dim() == 4:
                results.append(torch.mul(tensor, 2))
            else:
                results.append(tensor)
        return results
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[[10, 20], [30, 40]]]), torch.tensor([[[[1]]]]), torch.tensor([[1, 2], [3, 4]])]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    for (i, tensor) in enumerate(output):
        print(f'Output tensor {i}:')
        print(tensor)