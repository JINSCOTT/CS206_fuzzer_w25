import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = input_tensor
        result = torch.add(result, 5)
        result = torch.mul(result, 2)
        result = torch.sub(result, 3)
        for i in range(5):
            result = torch.add(result, i)
        return result
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]]), torch.tensor([[[9.0, 10.0], [11.0, 12.0]]]), torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([1.0])]
if __name__ == '__main__':
    model = PtModule()
    for (i, input_tensor) in enumerate(input_tensors):
        output = model(input_tensor)
        print(f'Output for input tensor {torch.add(i, 1)}:\n{output}')