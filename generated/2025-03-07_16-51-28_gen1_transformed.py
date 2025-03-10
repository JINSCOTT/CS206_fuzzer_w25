import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = 0
        for i in range(input_tensor.size(0)):
            result += torch.sum(input_tensor[i])
        return result
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[9, 10, 11], [12, 13, 14]]], dtype=torch.float32), torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=torch.float32), torch.tensor([[[2], [3]], [[4], [5]], [[6], [7]], [[8], [9]]], dtype=torch.float32), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Output for input tensor:\n{input_tensor}\nResult: {output.item()}\n')