import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        add_result = torch.add(x, 2)
        sub_result = torch.sub(x, 1)
        mul_result = torch.mul(x, 3)
        div_result = torch.div(x, 4)
        greater_than_result = x > 1
        less_than_result = x < 5
        return (add_result, sub_result, mul_result, div_result, greater_than_result, less_than_result)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]), torch.tensor([[5.0], [6.0]]), torch.tensor([10.0])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input tensor:\n{input_tensor}\nOutputs:\n{output}\n')