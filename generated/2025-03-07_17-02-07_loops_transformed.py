import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = torch.add(x, 2)
        result = torch.mul(result, 3)
        for i in range(result.size(0)):
            for j in range(result.size(1)):
                result[i, j] = torch.sub(result[i, j], 1)
        return result
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[1, 2, 3]], [[4, 5, 6]]], dtype=torch.float32), torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32), torch.tensor([[5, 6], [7, 8]], dtype=torch.float32), torch.tensor([[[[9]], [[10]]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')