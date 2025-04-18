import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = []
        for i in range(x.size(0)):
            temp = torch.mul(x[i], 2)
            temp = torch.add(temp, 3)
            temp = torch.div(temp, 2)
            result.append(temp)
        return torch.stack(result)
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[9, 10, 11], [12, 13, 14]], dtype=torch.float32), torch.tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=torch.float32), torch.tensor([[15], [16], [17]], dtype=torch.float32), torch.tensor([[[18]], [[19]], [[20]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')