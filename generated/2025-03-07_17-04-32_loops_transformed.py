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
            tensor_prod = torch.prod(tensor)
            results.append((tensor_sum.item(), tensor_mean.item(), tensor_prod.item()))
        return results
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[10, 20, 30], [40, 50, 60]]), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]]), torch.tensor([[[[1, 2, 3]], [[4, 5, 6]]]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)