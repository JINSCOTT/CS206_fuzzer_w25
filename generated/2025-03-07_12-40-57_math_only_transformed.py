import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.add(x, 2)
        x = torch.sub(x, 1)
        x = torch.mul(x, 3)
        x = torch.div(x, 4)
        return x
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])]
if __name__ == '__main__':
    model = PtModule()
    for (idx, input_tensor) in enumerate(input_tensors):
        output = model(input_tensor)
        print(f'Output for input tensor {torch.add(idx, 1)}:\n{output}')