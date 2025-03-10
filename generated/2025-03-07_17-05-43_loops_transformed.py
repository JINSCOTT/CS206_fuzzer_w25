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
            std_tensor = torch.std(tensor)
            results.append((sum_tensor, mean_tensor, std_tensor))
        return results
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[[[9], [10]], [[11], [12]]]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for (i, output) in enumerate(outputs):
        print(f'Output for tensor {i}: Sum={output[0]}, Mean={output[1]}, Std={output[2]}')