import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        output = torch.mul(x, 2)
        output = torch.add(output, 3)
        output = torch.sub(output, 1)
        output = torch.div(output, 2)
        for i in range(output.size(0)):
            for j in range(output.size(1)):
                output[i, j] = torch.pow(output[i, j], 2)
        return output
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[0, 1, 2], [3, 4, 5]]], dtype=torch.float32), torch.tensor([[1, 2, 3]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32), torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=torch.float32)]
if __name__ == '__main__':
    module = PtModule()
    for input_tensor in input_tensors:
        output_tensor = module(input_tensor)
        print(f'Input Tensor:\n{input_tensor}\nOutput Tensor:\n{output_tensor}\n')