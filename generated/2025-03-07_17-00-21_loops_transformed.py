import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            added = torch.add(tensor, 2)
            multiplied = torch.mul(tensor, 3)
            subtracted = torch.sub(tensor, 1)
            divided = torch.div(tensor, 2)
            results.append((added, multiplied, subtracted, divided))
        return results
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]), torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for (i, output) in enumerate(outputs):
        print(f'Output for tensor {i}:')
        for operation in output:
            print(operation)