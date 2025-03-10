import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        addition = torch.add(x, 1)
        subtraction = torch.sub(x, 1)
        multiplication = torch.mul(x, 2)
        division = torch.div(x, 2)
        comparison_greater = x > 0
        comparison_less = x < 0
        comparison_equal = x == 0
        return (addition, subtraction, multiplication, division, comparison_greater, comparison_less, comparison_equal)
input_tensors = [torch.tensor([[[1.0, -1.0, 3.0], [4.0, 5.0, -2.0]]]), torch.tensor([[[0.0, 2.0], [-3.0, 4.0]]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]), torch.tensor([[[-1.0], [0.0], [1.0]]]), torch.tensor([[[2.0, 3.0, -1.0, 5.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for (i, input_tensor) in enumerate(input_tensors):
        result = model(input_tensor)
        print(f'Results for input tensor {i}: {result}')