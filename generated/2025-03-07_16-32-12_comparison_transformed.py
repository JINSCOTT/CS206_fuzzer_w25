import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = {}
        results['sum'] = torch.add(inputs[0], inputs[1])
        results['difference'] = torch.sub(inputs[2], inputs[3])
        results['product'] = torch.mul(inputs[1], inputs[4])
        results['quotient'] = torch.div(inputs[4], torch.add(inputs[1], 1e-05))
        results['power'] = torch.pow(inputs[0], 2)
        results['greater_than'] = inputs[0] > inputs[1]
        results['less_than'] = inputs[2] < inputs[3]
        results['equal'] = inputs[1] == inputs[4]
        return results
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[5, 6], [7, 8]], dtype=torch.float32), torch.tensor([[9, 10, 11], [12, 13, 14]], dtype=torch.float32), torch.tensor([[15, 16], [17, 18]], dtype=torch.float32), torch.tensor([[19], [20]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)