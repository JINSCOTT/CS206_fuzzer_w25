import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = {}
        result['add'] = torch.add(x, 2)
        result['subtract'] = torch.sub(x, 1)
        result['multiply'] = torch.mul(x, 3)
        result['divide'] = torch.div(x, 4)
        loop_result = []
        for i in range(x.shape[0]):
            loop_result.append(torch.mul(x[i], i))
        result['loop'] = torch.stack(loop_result)
        return result
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[[9.0, 10.0], [11.0, 12.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for x in input_tensors:
        output = model(x)
        print(output)