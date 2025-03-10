import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        addition = torch.add(x, 5)
        subtraction = torch.sub(x, 3)
        multiplication = torch.mul(x, 2)
        division = torch.div(x, 4)
        power = torch.pow(x, 2)
        greater_than = x > 2
        less_than = x < 4
        equal_to = x == 1
        return {'addition': addition, 'subtraction': subtraction, 'multiplication': multiplication, 'division': division, 'power': power, 'greater_than': greater_than, 'less_than': less_than, 'equal_to': equal_to}
input_tensors = [torch.tensor([1, 2, 3], dtype=torch.float32), torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32), torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)]

def main():
    model = PtModule()
    for (idx, input_tensor) in enumerate(input_tensors):
        print(f'Input Tensor {torch.add(idx, 1)}:')
        output = model(input_tensor)
        for (key, value) in output.items():
            print(f'{key}: \n{value}\n')
if __name__ == '__main__':
    main()