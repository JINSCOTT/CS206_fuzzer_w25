import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        output = []
        for tensor in inputs:
            tensor_sum = torch.sum(tensor)
            tensor_mean = torch.mean(tensor)
            tensor_max = torch.max(tensor)
            output.append((tensor_sum, tensor_mean, tensor_max))
        return output
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1, 2, 3], [4, 5, 6]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[[-1, -2, -3], [-4, -5, -6]]]), torch.tensor([[0.0, 0.0], [0.0, 0.0]])]
if __name__ == '__main__':
    model = PtModule()
    results = model(input_tensors)
    for result in results:
        print(result)