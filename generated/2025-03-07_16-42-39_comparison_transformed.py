import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        addition = torch.add(inputs[0], inputs[1])
        subtraction = torch.sub(inputs[2], inputs[3])
        multiplication = torch.mul(inputs[1], inputs[2])
        division = torch.div(inputs[3], torch.add(inputs[4], 1e-08))
        comparison = inputs[0] > inputs[4]
        return (addition, subtraction, multiplication, division, comparison)
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[5], [6]], [[7], [8]]], dtype=torch.float32), torch.tensor([[9, 10, 11], [12, 13, 14]], dtype=torch.float32), torch.tensor([[15, 16], [17, 18]], dtype=torch.float32), torch.tensor([[[19], [20]], [[21], [22]]], dtype=torch.float32)]
if __name__ == '__main__':
    module = PtModule()
    results = module(input_tensors)
    for res in results:
        print(res)