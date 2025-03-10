import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result_add = torch.add(input_tensor, 2)
        result_sub = torch.sub(input_tensor, 1)
        result_mul = torch.mul(input_tensor, 3)
        result_div = torch.div(input_tensor, 2)
        for i in range(3):
            result_add += i
        return (result_add, result_sub, result_mul, result_div)
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32), torch.tensor([[[[1, 2]], [[3, 4]]]], dtype=torch.float32), torch.tensor([[[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=torch.float32), torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)