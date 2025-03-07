import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        add_result = torch.add(x, 5)
        subtract_result = torch.sub(x, 3)
        multiply_result = torch.mul(x, 2)
        divide_result = torch.div(x, 2)
        greater_than_result = x > 2
        return (add_result, subtract_result, multiply_result, divide_result, greater_than_result)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]), torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]), torch.tensor([[10.0, 20.0], [30.0, 40.0]]), torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print('Input Tensor:\n', input_tensor)
        print('Outputs:\n', output)