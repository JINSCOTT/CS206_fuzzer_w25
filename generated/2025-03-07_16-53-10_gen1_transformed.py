import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            processed_tensor = torch.mul(tensor, 2)
            results.append(processed_tensor)
        return results
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]), torch.tensor([[[1, 2, 3], [4, 5, 6]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])]
if __name__ == '__main__':
    module = PtModule()
    output = module(input_tensors)
    for (i, out) in enumerate(output):
        print(f'Output tensor {torch.add(i, 1)}:\n{out}\n')