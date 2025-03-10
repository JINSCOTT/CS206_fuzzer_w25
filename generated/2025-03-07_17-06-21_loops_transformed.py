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
            tensor_square = torch.pow(tensor, 2)
            results.append((tensor_sum, tensor_mean, tensor_square))
        return results
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1, 2, 3], [4, 5, 6]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), torch.tensor([1.0, 2.0, 3.0])]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)