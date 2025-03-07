import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        shape = input_tensor.shape
        output_tensor = torch.zeros_like(input_tensor)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    output_tensor[i, j, k] = torch.add(torch.mul(input_tensor[i, j, k], 2), 3)
        return output_tensor
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=torch.float32), torch.tensor([[[17, 18]]], dtype=torch.float32), torch.tensor([[[19, 20], [21, 22]], [[23, 24], [25, 26]]], dtype=torch.float32), torch.tensor([[[27, 28], [29, 30]], [[31, 32], [33, 34]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')