import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        output = []
        for tensor in x:
            added = torch.add(tensor, 1)
            multiplied = torch.mul(added, 2)
            output.append(multiplied)
        return output
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1], [2]], [[3], [4]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[7, 8], [9, 10]]), torch.tensor([[[2], [4]], [[6], [8]]])]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    for (i, out) in enumerate(output):
        print(f'Output for input tensor {torch.add(i, 1)}:\n{out}')