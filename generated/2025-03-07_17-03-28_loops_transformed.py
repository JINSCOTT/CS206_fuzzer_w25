import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = torch.add(x, 2)
        result = torch.mul(result, 3)
        for i in range(3):
            result = torch.sub(result, i)
        return result
input_tensors = [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]), torch.tensor([[5, 6, 7], [8, 9, 10]]), torch.tensor([[[[1], [2]], [[3], [4]], [[5], [6]]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor.float())
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')