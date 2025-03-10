import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output_tensor = torch.add(input_tensor, 2)
        output_tensor = torch.mul(output_tensor, 3)
        for i in range(output_tensor.shape[0]):
            for j in range(output_tensor.shape[1]):
                output_tensor[i, j] = torch.div(output_tensor[i, j], 2)
        return output_tensor
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[9, 10], [11, 12]]), torch.tensor([[[13], [14]], [[15], [16]]]), torch.tensor([[[1, 2, 3], [4, 5, 6]]])]
if __name__ == '__main__':
    model = PtModule()
    for inp in input_tensors:
        output = model(inp)
        print(f'Input:\n{inp}\nOutput:\n{output}\n')