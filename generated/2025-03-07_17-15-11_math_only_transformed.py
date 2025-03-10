import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        y1 = torch.add(x, 2)
        y2 = torch.sub(y1, 1)
        y3 = torch.mul(y2, 3)
        y4 = torch.div(y3, 4.0)
        y5 = torch.pow(y4, 2)
        return y5
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32), torch.tensor([[[[1, 2]], [[3, 4]]]], dtype=torch.float32), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32), torch.tensor([[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for (i, input_tensor) in enumerate(input_tensors):
        output = model(input_tensor)
        print(f'Output for input tensor {torch.add(i, 1)}:\n{output}')