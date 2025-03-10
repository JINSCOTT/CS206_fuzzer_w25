import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output = torch.mul(input_tensor, 2)
        output = torch.add(output, 3)
        output = torch.sub(output, 1)
        output = torch.div(output, 4)
        for i in range(5):
            output = torch.add(output, i)
        return output
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[1], [2], [3], [4]]), torch.tensor([[[1]], [[2]], [[3]], [[4]]]), torch.tensor([1, 2, 3, 4])]
if __name__ == '__main__':
    model = PtModule()
    for (i, tensor) in enumerate(input_tensors):
        output = model(tensor)
        print(f'Output for input tensor {i}:\n{output}')