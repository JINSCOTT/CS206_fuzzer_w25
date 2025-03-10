import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            result = torch.pow(input_tensor, 2)
            results.append(result)
        return results
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=torch.float32), torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=torch.float32), torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]], [[9, 10, 11, 12]]], dtype=torch.float32), torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]], [[[5]], [[6]]]], dtype=torch.float32)]
if __name__ == '__main__':
    module = PtModule()
    output = module(input_tensors)
    print(output)