import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        addition = torch.add(x, 2)
        subtraction = torch.sub(x, 2)
        multiplication = torch.mul(x, 2)
        division = torch.div(x, 2)
        comparison = x > 1
        return (addition, subtraction, multiplication, division, comparison)
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=torch.float32), torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        result = model(input_tensor)
        print(result)