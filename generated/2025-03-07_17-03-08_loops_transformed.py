import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        x = torch.add(input_tensor, 2)
        y = torch.mul(x, 3)
        z = torch.sub(y, 5)
        result = torch.div(z, 4)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = torch.pow(result[i, j], 2)
        return result
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=torch.float32), torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32), torch.tensor([[[1]], [[2]], [[3]], [[4]]], dtype=torch.float32), torch.tensor([[[1, 2], [3, 4], [5, 6]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)