import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            processed = torch.add(torch.mul(tensor, 2), 10)
            results.append(processed)
        return results
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=torch.float32), torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32), torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], dtype=torch.float32), torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for (i, output) in enumerate(outputs):
        print(f'Output {i}:')
        print(output)