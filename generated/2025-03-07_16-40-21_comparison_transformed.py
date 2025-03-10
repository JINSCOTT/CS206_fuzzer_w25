import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        results = {}
        results['add'] = torch.add(x, 2)
        results['subtract'] = torch.sub(x, 2)
        results['multiply'] = torch.mul(x, 2)
        results['divide'] = torch.div(x, 2)
        results['greater_than'] = x > 1
        results['less_than'] = x < 3
        results['equal_to'] = x == 2
        return results
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[1.5, 2.5], [3.5, 4.5]]), torch.tensor([1, 2, 3, 4, 5])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')