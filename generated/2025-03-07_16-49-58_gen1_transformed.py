import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = torch.add(torch.mul(tensor, 2), 1)
            results.append(result)
        return results
input_tensors = [torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32), torch.tensor([[13, 14, 15, 16], [17, 18, 19, 20]], dtype=torch.float32), torch.tensor([[[21, 22], [23, 24], [25, 26]]], dtype=torch.float32), torch.tensor([[[27], [28]], [[29], [30]], [[31], [32]]], dtype=torch.float32), torch.tensor([[[33, 34, 35]], [[36, 37, 38]], [[39, 40, 41]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    for (idx, tensor) in enumerate(output):
        print(f'Output tensor {idx}:')
        print(tensor)