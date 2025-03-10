import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output_tensor = torch.mul(input_tensor, 2)
        for i in range(1, 4):
            output_tensor += torch.div(input_tensor, i)
        return output_tensor
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[9, 10, 11], [12, 13, 14]], [[15, 16, 17], [18, 19, 20]]], dtype=torch.float32), torch.tensor([[[21, 22], [23, 24], [25, 26]]], dtype=torch.float32), torch.tensor([[[27, 28, 29, 30], [31, 32, 33, 34]]], dtype=torch.float32), torch.tensor([[[35], [36]], [[37], [38]], [[39], [40]]], dtype=torch.float32)]
if __name__ == '__main__':
    pt_module = PtModule()
    for (i, input_tensor) in enumerate(input_tensors):
        output = pt_module(input_tensor)
        print(f'Output for input tensor {torch.add(i, 1)}:\n{output}\n')