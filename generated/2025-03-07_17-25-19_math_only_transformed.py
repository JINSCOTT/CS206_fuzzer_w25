import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        x = torch.add(x, 2)
        x = torch.sub(x, 3)
        x = torch.mul(x, 4)
        x = torch.div(x, 5)
        return x
input_tensors = [torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32), torch.tensor([[2, 3], [5, 7]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32), torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for (i, tensor) in enumerate(input_tensors):
        output = model(tensor)
        print(f'Output for input tensor {i}:\n{output}')