import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            tensor_sum = torch.sum(tensor)
            tensor_mean = torch.mean(tensor)
            tensor_max = torch.max(tensor)
            tensor_min = torch.min(tensor)
            results.append((tensor_sum, tensor_mean, tensor_max, tensor_min))
        return results
input_tensors = [torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]), torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]), torch.tensor([[[1, 2, 3]], [[4, 5, 6]]]), torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]])]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)