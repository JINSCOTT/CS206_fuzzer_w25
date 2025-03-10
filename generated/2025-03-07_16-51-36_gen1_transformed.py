import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output_tensor = torch.mul(input_tensor, 2)
        for i in range(output_tensor.size(0)):
            output_tensor[i] = torch.add(output_tensor[i], 1)
        return output_tensor
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]), torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])]
if __name__ == '__main__':
    pt_module = PtModule()
    for input_tensor in input_tensors:
        output_tensor = pt_module(input_tensor)
        print('Output Tensor:', output_tensor)