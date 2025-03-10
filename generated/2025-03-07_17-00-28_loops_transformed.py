import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = []
        for i in range(x.size(0)):
            batch_result = []
            for j in range(x.size(1)):
                value = x[i, j]
                squared = torch.pow(value, 2)
                doubled = torch.mul(value, 2)
                summed = torch.add(squared, doubled)
                batch_result.append(summed)
            result.append(batch_result)
        return torch.tensor(result)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Input:\n{tensor}\nOutput:\n{output}\n')