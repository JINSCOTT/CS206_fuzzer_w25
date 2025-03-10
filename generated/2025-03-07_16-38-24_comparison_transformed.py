import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result1 = torch.add(x, 5)
        result2 = torch.sub(x, 3)
        result3 = torch.mul(x, 2)
        result4 = torch.div(x, 2)
        result5 = torch.pow(x, 2)
        result6 = x > 0
        result7 = x < 10
        result8 = x == 5
        return (result1, result2, result3, result4, result5, result6, result7, result8)
input_tensors = [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Output for input tensor:\n{input_tensor}\n{output}\n')