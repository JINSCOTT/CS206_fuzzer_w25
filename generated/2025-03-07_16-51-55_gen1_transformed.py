import torch

class PtModule:

    def __init__(self):
        pass

    def forward(self, input_tensors):
        results = []
        for tensor in input_tensors:
            result = torch.add(tensor, 5)
            results.append(result)
        return results
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([1, 2, 3, 4]), torch.tensor([[[[1], [2]], [[3], [4]]]]), torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])]
if __name__ == '__main__':
    module = PtModule()
    outputs = module.forward(input_tensors)
    for (i, output) in enumerate(outputs):
        print(f'Output {torch.add(i, 1)}:')
        print(output)