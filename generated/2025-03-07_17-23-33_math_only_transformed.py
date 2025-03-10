import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        result = []
        for input_tensor in inputs:
            added = torch.add(input_tensor, 2)
            subtracted = torch.sub(input_tensor, 1)
            multiplied = torch.mul(input_tensor, 3)
            divided = torch.div(input_tensor, 2)
            result.append((added, subtracted, multiplied, divided))
        return result
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[7, 8, 9]]), torch.tensor([[[10], [11]], [[12], [13]]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for (i, output) in enumerate(outputs):
        print(f'Output for input tensor {i}:\n{output}\n')