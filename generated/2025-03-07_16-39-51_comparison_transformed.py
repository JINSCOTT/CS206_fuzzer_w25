import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        addition = torch.add(x, 2)
        subtraction = torch.sub(x, 3)
        multiplication = torch.mul(x, 4)
        division = torch.div(x, 5)
        power = torch.pow(x, 2)
        greater_than = x > 1
        less_than = x < 2
        equal_to = x == 0
        return {'addition': addition, 'subtraction': subtraction, 'multiplication': multiplication, 'division': division, 'power': power, 'greater_than': greater_than, 'less_than': less_than, 'equal_to': equal_to}
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([1, 2, 3, 4]), torch.tensor([[1, 2], [3, 4], [5, 6]]), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]])]

def main():
    model = PtModule()
    for (i, input_tensor) in enumerate(input_tensors):
        print(f'Input Tensor {torch.add(i, 1)}:')
        output = model(input_tensor)
        for (key, value) in output.items():
            print(f'{key}: {value}')
if __name__ == '__main__':
    main()