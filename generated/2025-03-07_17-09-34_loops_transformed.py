import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output_tensor = torch.mul(input_tensor, 2)
        output_tensor = torch.add(output_tensor, 3)
        for i in range(output_tensor.size(0)):
            output_tensor[i] = torch.sub(output_tensor[i], 1)
        output_tensor = torch.pow(output_tensor, 2)
        return output_tensor
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[[1], [2]], [[3], [4]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')