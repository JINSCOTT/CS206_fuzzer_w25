import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            tensor_sum = input_tensor.sum()
            tensor_mean = input_tensor.mean()
            results.append((tensor_sum, tensor_mean))
        return results
input_tensors = [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]], [[[5.0], [6.0]], [[7.0], [8.0]]]]), torch.tensor([[[-1.0, -2.0], [-3.0, -4.0]], [[-5.0, -6.0], [-7.0, -8.0]]]), torch.tensor([[[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]]])]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)