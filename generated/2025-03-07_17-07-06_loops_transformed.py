import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result_add = torch.add(x, 2)
        result_mul = torch.mul(x, 3)
        for i in range(x.size(0)):
            result_add[i] = torch.add(result_add[i], i)
        mean_result = result_add.mean()
        return (result_add, result_mul, mean_result)
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32), torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]], dtype=torch.float32), torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input Tensor:\n{input_tensor}\nOutput:\n{output}\n')