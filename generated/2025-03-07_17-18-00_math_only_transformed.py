import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2, x3, x4, x5):
        add = torch.add(x1, x2)
        sub = torch.sub(x3, x4)
        mul = torch.mul(x1, x5)
        div = torch.div(x4, torch.add(x5, 1e-05))
        power = torch.pow(x2, 2)
        return (add, sub, mul, div, power)
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=torch.float32), torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[5, 6], [7, 8]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    results = model(*input_tensors)
    for result in results:
        print(result)