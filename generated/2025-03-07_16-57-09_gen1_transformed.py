import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = input_tensor.clone()
        for i in range(result.shape[0]):
            result[i] = torch.add(result[i], i)
        return result
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32), torch.tensor([[[10, 20], [30, 40], [50, 60]], [[70, 80], [90, 100], [110, 120]]], dtype=torch.float32), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]], dtype=torch.float32)]
if __name__ == '__main__':
    module = PtModule()
    for input_tensor in input_tensors:
        output = module(input_tensor)
        print('Output for input tensor:\n', output)