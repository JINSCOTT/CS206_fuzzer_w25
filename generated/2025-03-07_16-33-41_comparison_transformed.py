import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2):
        add_result = torch.add(x1, x2)
        sub_result = torch.sub(x1, x2)
        mul_result = torch.mul(x1, x2)
        div_result = torch.div(x1, torch.add(x2, 1e-08))
        greater_result = x1 > x2
        less_result = x1 < x2
        return (add_result, sub_result, mul_result, div_result, greater_result, less_result)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[1, 2], [3, 4]]), torch.tensor([1, 2, 3, 4, 5])]
if __name__ == '__main__':
    model = PtModule()
    results = model(input_tensors[0], input_tensors[0])
    print('Results:', results)