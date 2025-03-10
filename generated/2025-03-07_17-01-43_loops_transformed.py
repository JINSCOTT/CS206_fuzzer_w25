import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            sum_tensor = torch.sum(tensor)
            mean_tensor = torch.mean(tensor)
            results.append((sum_tensor.item(), mean_tensor.item()))
        return results
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]), torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), torch.tensor([[5.0], [6.0], [7.0], [8.0]])]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)