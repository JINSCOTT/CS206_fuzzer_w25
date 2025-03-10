import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = []
        for tensor in x:
            added = torch.add(tensor, 10)
            multiplied = torch.mul(added, 2)
            squared = torch.pow(multiplied, 2)
            result.append(squared)
        return result
input_tensors = [torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32), torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32), torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32), torch.tensor([[10, 20], [30, 40], [50, 60], [70, 80]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)