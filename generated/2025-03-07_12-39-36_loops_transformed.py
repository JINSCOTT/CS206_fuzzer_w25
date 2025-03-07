import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output = torch.mul(input_tensor, 2)
        output = torch.add(output, 5)
        for i in range(2):
            output = torch.sub(output, torch.add(i, 1))
        output = torch.div(output, 3)
        return output
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([1, 2, 3]), torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])]
if __name__ == '__main__':
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f'Output for input tensor {i}: {output}')