import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        addition = torch.add(inputs[0], inputs[1])
        subtraction = torch.sub(inputs[1], inputs[2])
        multiplication = torch.mul(inputs[2], inputs[3])
        division = torch.div(inputs[3], torch.add(inputs[4], 1e-05))
        comparison = inputs[0] > inputs[4]
        return (addition, subtraction, multiplication, division, comparison)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[9.0, 10.0]], [[11.0, 12.0]]]]), torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[0.0, 1.0], [2.0, 3.0]])]
if __name__ == '__main__':
    model = PtModule()
    results = model(input_tensors)
    print('Results:')
    for res in results:
        print(res)