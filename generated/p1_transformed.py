import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.mul(x, 2.5)
        x = torch.sub(x, torch.mean(x))
        x = torch.exp(x)
        x = torch.div(x, torch.add(torch.abs(x), 1))
        x = torch.matmul(x, x.transpose(-1, -2)) if x.dim() == 3 else x
        x = torch.sqrt(x)
        return x
input_tensors = [torch.tensor([[[0.5, -1.2, 0.3], [1.5, 0.2, -0.8], [-0.5, 0.9, 1.1]], [[-0.3, 0.8, -1.5], [0.6, -0.7, 0.4], [1.2, 0.1, -0.9]], [[0.7, -0.6, 0.5], [-1.1, 0.3, 0.2], [0.4, -0.2, 0.9]]]), torch.tensor([[[[0.2, -0.5, 0.6, -0.1], [0.7, 0.3, -0.8, 0.9], [-0.4, 1.0, -0.2, 0.5], [0.6, -0.7, 0.8, -0.3]], [[-0.6, 0.7, -0.9, 0.4], [0.5, -0.2, 0.3, -0.8], [0.9, 0.1, -0.5, 0.6], [-0.3, 0.8, -0.7, 0.2]]]])]
if __name__ == '__main__':
    model = PtModule()
    sample_input = input_tensors[0]
    output = model(sample_input)
    print('Sample Input Shape:', sample_input.shape)
    print('Output Shape:', output.shape)
    print('Output Tensor:\n', output)