import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.add(x, 5)
        x = torch.sub(x, 2)
        x = torch.mul(x, 3)
        x = torch.div(x, 2)
        return x
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output_tensor = model(input_tensor)
        print('Input Tensor:\n', input_tensor)
        print('Output Tensor:\n', output_tensor)