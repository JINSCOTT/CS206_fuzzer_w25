import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        add_result = torch.add(x, 2)
        sub_result = torch.sub(x, 1)
        mul_result = torch.mul(x, 3)
        div_result = torch.div(x, 2)
        greater_than = x > 1
        less_than = x < 5
        equal_to = x == 3
        return (add_result, sub_result, mul_result, div_result, greater_than, less_than, equal_to)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]), torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])]
if __name__ == '__main__':
    model = PtModule()
    for (idx, input_tensor) in enumerate(input_tensors):
        output = model(input_tensor)
        print(f'Output for input tensor {idx}:\n{output}\n')