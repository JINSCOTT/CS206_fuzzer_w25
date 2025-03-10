import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        add_result = torch.add(x, 2)
        subtract_result = torch.sub(x, 1)
        multiply_result = torch.mul(x, 3)
        divide_result = torch.div(x, 2)
        greater_than = x > 1
        less_than = x < 5
        return (add_result, subtract_result, multiply_result, divide_result, greater_than, less_than)
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[9, 10], [11, 12]]]), torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]), torch.tensor([[[5], [10]], [[15], [20]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')