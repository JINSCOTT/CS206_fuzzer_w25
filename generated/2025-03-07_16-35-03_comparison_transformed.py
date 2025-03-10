import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        sum_result = torch.add(x, 5)
        sub_result = torch.sub(x, 2)
        mul_result = torch.mul(x, 3)
        div_result = torch.div(x, 2)
        greater_than = x > 3
        less_than = x < 4
        return {'sum_result': sum_result, 'sub_result': sub_result, 'mul_result': mul_result, 'div_result': div_result, 'greater_than': greater_than, 'less_than': less_than}
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[9, 10, 11], [12, 13, 14]]], dtype=torch.float32), torch.tensor([[[15], [16]], [[17], [18]], [[19], [20]]], dtype=torch.float32), torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]], dtype=torch.float32), torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for (i, input_tensor) in enumerate(input_tensors):
        print(f'Input Tensor {i} Results:')
        output = model(input_tensor)
        for (key, value) in output.items():
            print(f'{key}: {value}')