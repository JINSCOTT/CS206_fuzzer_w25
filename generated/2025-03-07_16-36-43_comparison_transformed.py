import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        add_result = torch.add(x, 2)
        sub_result = torch.sub(x, 2)
        mul_result = torch.mul(x, 2)
        div_result = torch.div(x, 2)
        greater_than = x > 1
        less_than = x < 3
        equal_to = x == 2
        return (add_result, sub_result, mul_result, div_result, greater_than, less_than, equal_to)
input_tensors = [torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32), torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32), torch.tensor([[1]], dtype=torch.float32), torch.tensor([[[[1], [2]], [[3], [4]]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for x in input_tensors:
        results = model(x)
        print(results)