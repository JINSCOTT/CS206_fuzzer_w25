import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            tensor = torch.add(tensor, 2)
            tensor = torch.mul(tensor, 3)
            tensor = torch.div(tensor, 2)
            tensor = torch.sub(tensor, 1)
            results.append(tensor)
        return results
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32), torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[[1, 2]], [[3, 4]]]], dtype=torch.float32), torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    for (i, out) in enumerate(output):
        print(f'Output for tensor {i}: {out}')