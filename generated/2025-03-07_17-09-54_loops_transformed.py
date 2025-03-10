import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        y = torch.add(x, 2)
        y = torch.mul(y, 3)
        y = torch.sub(y, 1)
        y = torch.div(y, 4)
        (batch_size, channels, height, width) = y.shape
        for i in range(batch_size):
            for j in range(channels):
                y[i, j] = torch.pow(y[i, j], 2)
        return y
input_tensors = [torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32), torch.tensor([[[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[9, 10], [11, 12]]], dtype=torch.float32), torch.tensor([[[13, 14], [15, 16]]], dtype=torch.float32), torch.tensor([[[17, 18], [19, 20]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)