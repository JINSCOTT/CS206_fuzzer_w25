import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output_tensor = input_tensor.clone()
        for i in range(input_tensor.shape[0]):
            output_tensor[i] = torch.add(torch.mul(output_tensor[i], 2), 1)
        return output_tensor
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=torch.float32), torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32), torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Input Tensor:\n{tensor}\nOutput Tensor:\n{output}\n')