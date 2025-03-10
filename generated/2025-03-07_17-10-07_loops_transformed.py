import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensors):
        results = []
        for tensor in input_tensors:
            tensor = torch.add(tensor, 2)
            tensor = torch.mul(tensor, 3)
            tensor = torch.sub(tensor, 1)
            results.append(tensor)
        return results
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], dtype=torch.float32), torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32), torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)