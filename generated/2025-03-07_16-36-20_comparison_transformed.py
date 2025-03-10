import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        results = {}
        results['addition'] = torch.add(x, 5)
        results['subtraction'] = torch.sub(x, 2)
        results['multiplication'] = torch.mul(x, 3)
        results['division'] = torch.div(x, 4)
        results['greater_than'] = x > 10
        results['less_than_equal'] = x <= 5
        return results
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[6, 7, 8], [9, 10, 11]], dtype=torch.float32), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[[3, 4], [5, 6]]]], dtype=torch.float32), torch.tensor([[10, 11, 12], [13, 14, 15]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')