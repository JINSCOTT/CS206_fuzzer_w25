import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            mean_value = torch.mean(tensor)
            results.append(mean_value)
        return results
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32), torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32)]
if __name__ == '__main__':
    module = PtModule()
    outputs = module(input_tensors)
    print(outputs)