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
            tensor_product = torch.prod(tensor)
            output.append((tensor_sum, tensor_mean, tensor_product))
        return output
input_tensors = [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]), torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])]
if __name__ == '__main__':
    model = PtModule()
    results = model(input_tensors)
    for (idx, result) in enumerate(results):
        print(f'Tensor {torch.add(idx, 1)}: Sum = {result[0]}, Mean = {result[1]}, Product = {result[2]}')