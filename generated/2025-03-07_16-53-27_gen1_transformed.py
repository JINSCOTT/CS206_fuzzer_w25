import torch

class PtModule(torch.nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = torch.add(torch.mul(tensor, 2), 3)
            results.append(result)
        return results
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]), torch.tensor([[7, 8, 9], [10, 11, 12]]), torch.tensor([[[10, 11], [12, 13]], [[14, 15], [16, 17]]])]
if __name__ == '__main__':
    module = PtModule()
    output = module(input_tensors)
    for (i, out) in enumerate(output):
        print(f'Output for tensor {torch.add(i, 1)}:\n{out}')