import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        (a, b, c, d, e) = inputs
        add_result = torch.add(a, b)
        sub_result = torch.sub(b, c)
        mul_result = torch.mul(c, d)
        div_result = torch.div(d, torch.add(e, 1e-05))
        greater_than = a > b
        less_than = b < c
        equal_to = c == d
        return (add_result, sub_result, mul_result, div_result, greater_than, less_than, equal_to)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[5.0], [6.0]], [[7.0], [8.0]]]), torch.tensor([1.0]), torch.tensor([[10.0, 20.0]]), torch.tensor([0.0])]
if __name__ == '__main__':
    model = PtModule()
    results = model(input_tensors)
    for result in results:
        print(result)