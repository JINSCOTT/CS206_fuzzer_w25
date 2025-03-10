import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output = torch.add(input_tensor, 2)
        output = torch.mul(output, 3)
        output = torch.sub(output, 1)
        output = torch.div(output, 2)
        for i in range(output.size(0)):
            output[i] = torch.pow(output[i], 2)
        return output
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[1.5, 2.5], [3.5, 4.5]]), torch.tensor([[[1], [2]], [[3], [4]]]), torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')