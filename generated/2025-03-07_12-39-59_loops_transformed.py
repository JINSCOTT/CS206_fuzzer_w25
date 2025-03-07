import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = torch.empty_like(x)
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                result[i, j] = torch.add(torch.mul(x[i, j], 2), 3)
                if result[i, j].sum() > 10:
                    result[i, j] = torch.sub(result[i, j], 5)
        return result
input_tensors = [torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]]]]), torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]], [[[10.0, 12.0], [14.0, 16.0]]], [[[18.0, 20.0], [22.0, 24.0]]]]), torch.tensor([[[[1.5, 2.5], [3.5, 4.5]]], [[[5.5, 6.5], [7.5, 8.5]]], [[[9.5, 10.5], [11.5, 12.5]]]]), torch.tensor([[[[3.0, 6.0], [9.0, 12.0]]], [[[15.0, 18.0], [21.0, 24.0]]], [[[27.0, 30.0], [33.0, 36.0]]]]), torch.tensor([[[[0.5, 1.5], [2.5, 3.5]]], [[[4.5, 5.5], [6.5, 7.5]]], [[[8.5, 9.5], [10.5, 11.5]]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)