import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = {}
        results['add'] = torch.add(inputs[0], inputs[1])
        results['subtract'] = torch.sub(inputs[1], inputs[0])
        results['multiply'] = torch.mul(inputs[0], inputs[2])
        results['divide'] = torch.div(inputs[2], torch.add(inputs[3], 1e-05))
        results['greater_than'] = inputs[0] > inputs[1]
        results['less_than'] = inputs[1] < inputs[2]
        return results
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[5, 6], [7, 8]], dtype=torch.float32), torch.tensor([[2, 3], [4, 5]], dtype=torch.float32), torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    results = model(input_tensors)
    print(results)