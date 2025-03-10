import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x1 = torch.add(x, 2)
        x2 = torch.sub(x, 1)
        x3 = torch.mul(x, 3)
        x4 = torch.div(x, 2)
        return (x1, x2, x3, x4)
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=torch.float32), torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32), torch.tensor([[[1]], [[2]], [[3]], [[4]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)