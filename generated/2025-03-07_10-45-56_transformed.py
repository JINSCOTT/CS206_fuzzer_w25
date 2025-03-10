import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            if tensor.dim() == 3:
                result = torch.add(tensor, 2)
            elif tensor.dim() == 4:
                result = torch.mul(tensor, 1.5)
            else:
                result = tensor.pow(2)
            results.append(result)
        return results
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]]]]), torch.tensor([[1.0, 2.0], [3.0, 4.0]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for (i, output) in enumerate(outputs):
        print(f'Output {i}:')
        print(output)