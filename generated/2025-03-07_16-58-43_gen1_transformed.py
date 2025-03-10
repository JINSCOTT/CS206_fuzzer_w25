import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = []
        for tensor in x:
            processed_tensor = torch.add(torch.mean(tensor), torch.std(tensor))
            result.append(processed_tensor)
        return result
input_tensors = [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]), torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0], [7.0, 8.0]])]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)