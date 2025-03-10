import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        sum_result = torch.add(x, 2)
        sub_result = torch.sub(sum_result, 3)
        mul_result = torch.mul(sub_result, 4)
        div_result = torch.div(mul_result, 2)
        greater_than = div_result > 5
        less_than = div_result < 10
        equal_to = div_result == 6
        return (sum_result, sub_result, mul_result, div_result, greater_than, less_than, equal_to)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]]), torch.tensor([1.0, 2.0, 3.0]), torch.tensor([[[[1.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input: {input_tensor}\nOutput: {output}\n')