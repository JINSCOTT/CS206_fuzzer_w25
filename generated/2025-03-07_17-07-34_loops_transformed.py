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
            max_tensor = torch.max(tensor)
            min_tensor = torch.min(tensor)
            results.append((sum_tensor, mean_tensor, max_tensor, min_tensor))
        return results
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]]), torch.tensor([[[1, 1, 1], [2, 2, 2]]]), torch.tensor([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]], [[9.5, 10.5], [11.5, 12.5]]]), torch.tensor([[3, 1, 2], [4, 6, 5]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for (i, output) in enumerate(outputs):
        print(f'Output for tensor {i}: Sum: {output[0]}, Mean: {output[1]}, Max: {output[2]}, Min: {output[3]}')