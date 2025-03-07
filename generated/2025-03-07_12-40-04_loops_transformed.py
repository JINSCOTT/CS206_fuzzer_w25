import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for inp in inputs:
            output = torch.add(inp, 2)
            output = torch.mul(output, 3)
            output = torch.sub(output, 1)
            results.append(output)
        return results
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]]), torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[[2.0]], [[3.0]], [[4.0]]]), torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for i, output in enumerate(outputs):
        print(f'Output tensor {i}: {output}')