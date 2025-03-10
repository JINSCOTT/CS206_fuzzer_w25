import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        added = torch.add(input_tensor, 2)
        subtracted = torch.sub(input_tensor, 3)
        multiplied = torch.mul(input_tensor, 4)
        divided = torch.div(input_tensor, 5)
        greater_than = input_tensor > 1
        less_than_equal = input_tensor <= 2
        return (added, subtracted, multiplied, divided, greater_than, less_than_equal)
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]), torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]]), torch.tensor([[0.0, -1.0], [1.0, 5.0]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Output for tensor {tensor}: {output}')