import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        added = torch.add(x, 2)
        subtracted = torch.sub(x, 2)
        multiplied = torch.mul(x, 2)
        divided = torch.div(x, 2)
        greater_than = x > 1
        less_than = x < 1
        equal_to = x == 1
        return (added, subtracted, multiplied, divided, greater_than, less_than, equal_to)
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[5, 10, 15], [20, 25, 30]], dtype=torch.float32), torch.tensor([[[1, 0], [0, 1]], [[1, 1], [1, 1]]], dtype=torch.float32), torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for (i, input_tensor) in enumerate(input_tensors):
        output = model(input_tensor)
        print(f'Output for input tensor {i}:\n{output}\n')