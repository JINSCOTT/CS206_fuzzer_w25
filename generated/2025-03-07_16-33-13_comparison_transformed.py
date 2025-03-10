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
        gt_result = x > 2
        lt_result = x < 1
        eq_result = x == 0
        return (add_result, sub_result, mul_result, div_result, gt_result, lt_result, eq_result)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([2.0, 0.0, -1.0]), torch.tensor([[[3.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        (add, sub, mul, div, gt, lt, eq) = model(input_tensor)
        print(f'Input:\n{input_tensor}\n')
        print(f'Add Result:\n{add}\n')
        print(f'Subtract Result:\n{sub}\n')
        print(f'Multiply Result:\n{mul}\n')
        print(f'Divide Result:\n{div}\n')
        print(f'Greater Than Result:\n{gt}\n')
        print(f'Less Than Result:\n{lt}\n')
        print(f'Equality Result:\n{eq}\n')