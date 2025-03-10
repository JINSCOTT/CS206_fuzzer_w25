import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        results = {}
        results['addition'] = torch.add(x, 2)
        results['subtraction'] = torch.sub(x, 2)
        results['multiplication'] = torch.mul(x, 2)
        results['division'] = torch.div(x, 2)
        squared_values = torch.empty_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                squared_values[i][j] = torch.pow(x[i][j], 2)
        results['squared'] = squared_values
        return results
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), torch.tensor([1.0, 2.0, 3.0])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input tensor:\n{input_tensor}\nOutput:\n{output}\n')