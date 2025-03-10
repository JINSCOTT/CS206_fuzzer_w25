import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        add_result = torch.add(x, 5)
        sub_result = torch.sub(x, 3)
        mul_result = torch.mul(x, 2)
        div_result = torch.div(x, 4)
        comparison_result = x > 2
        return (add_result, sub_result, mul_result, div_result, comparison_result)
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32), torch.tensor([[5, 6], [7, 8]], dtype=torch.float32), torch.tensor([[10, 9, 8], [7, 6, 5]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print('Input Tensor:\n', input_tensor)
        print('Output:\n', output)
        print(torch.mul('-', 40))