import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        addition = torch.add(input_tensor, 5)
        subtraction = torch.sub(input_tensor, 2)
        multiplication = torch.mul(input_tensor, 3)
        division = torch.div(input_tensor, 2)
        power = torch.pow(input_tensor, 2)
        greater_than = input_tensor > 3
        less_than = input_tensor < 10
        equal_to = input_tensor == 5
        return {'addition': addition, 'subtraction': subtraction, 'multiplication': multiplication, 'division': division, 'power': power, 'greater_than': greater_than, 'less_than': less_than, 'equal_to': equal_to}
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[1, 5, 3], [7, 8, 9]]], dtype=torch.float32), torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.float32), torch.tensor([[[2], [3]], [[4], [5]], [[6], [7]]], dtype=torch.float32), torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]], [[9, 10, 11, 12]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for (i, tensor) in enumerate(input_tensors):
        print(f'Input Tensor {i} Results:')
        results = model(tensor)
        for (key, value) in results.items():
            print(f'{key}: \n{value}\n')