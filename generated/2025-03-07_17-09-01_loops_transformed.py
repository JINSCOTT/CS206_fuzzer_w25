import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            add_result = torch.add(input_tensor, 2)
            mul_result = torch.mul(input_tensor, 3)
            div_result = torch.div(input_tensor, 2)
            results.append((add_result, mul_result, div_result))
        return results
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[-1, -2, -3], [-4, -5, -6]]], dtype=torch.float32), torch.tensor([[[10, 20, 30, 40]]], dtype=torch.float32), torch.tensor([[[[1]]]], dtype=torch.float32), torch.tensor([[[0]], [[1]], [[2]], [[3]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)