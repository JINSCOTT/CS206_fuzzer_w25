import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        added = torch.add(x, 2)
        subtracted = torch.sub(x, 1)
        multiplied = torch.mul(x, 3)
        divided = torch.div(x, 2)
        less_than = x < 1
        greater_than = x > 1
        return (added, subtracted, multiplied, divided, less_than, greater_than)
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[1.0, 2.0]]), torch.tensor([[[[9.0]]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        results = model(tensor)
        print('Input Tensor:\n', tensor)
        print('Results:\n', results)