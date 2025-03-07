import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output = input_tensor.clone()
        for i in range(output.size(0)):
            output[i] = torch.add(output[i], 2)
            output[i] = torch.mul(output[i], 3)
        return output
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[10, 20, 30], [40, 50, 60]]), torch.tensor([[[1]], [[2]], [[3]], [[4]]]), torch.tensor([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]])]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        output = module(tensor)
        print(f'Input:\n{tensor}\nOutput:\n{output}\n')