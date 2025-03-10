import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = input_tensor.clone()
        result = torch.add(result, 2)
        result = torch.mul(result, 3)
        for i in range(result.size(-1)):
            result[..., i] = torch.sub(result[..., i], 1)
        return result
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32), torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.float32), torch.tensor([[1], [2], [3]], dtype=torch.float32), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')