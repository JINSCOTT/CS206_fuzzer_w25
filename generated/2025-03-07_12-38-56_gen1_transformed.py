import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = torch.add(torch.mean(tensor), torch.std(tensor))
            results.append(result)
        return results
input_tensors = [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]]), torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for i, output in enumerate(outputs):
        print(f'Output for tensor {i}: {output.item()}')