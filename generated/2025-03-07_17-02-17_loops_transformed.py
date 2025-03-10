import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            multiplied = torch.mul(input_tensor, 2)
            added = torch.add(multiplied, 3)
            results.append(added)
        return results
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1], [2]], [[3], [4]]]), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), torch.tensor([[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]]), torch.tensor([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for (i, output) in enumerate(outputs):
        print(f'Output for input tensor {i}: {output}')