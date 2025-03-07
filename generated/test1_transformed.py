import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            mean_value = torch.mean(input_tensor)
            sum_value = torch.sum(input_tensor)
            product_value = torch.prod(input_tensor)
            results.append((mean_value, sum_value, product_value))
        return results
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float32), torch.tensor([[[1, 2, 3]], [[4, 5, 6]]], dtype=torch.float32), torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)