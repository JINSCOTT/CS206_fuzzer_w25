import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            mean_val = input_tensor.mean()
            sum_val = input_tensor.sum()
            max_val = input_tensor.max()
            min_val = input_tensor.min()
            results.append((mean_val, sum_val, max_val, min_val))
        return results
input_tensors = [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[[1], [2]], [[3], [4]]]], [[[[5], [6]], [[7], [8]]]]]), torch.tensor([[9, 10, 11, 12], [13, 14, 15, 16]]), torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)