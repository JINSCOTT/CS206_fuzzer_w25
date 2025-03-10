import torch

class PtModule:

    def __init__(self):
        pass

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = torch.add(torch.mul(tensor, 2), 5)
            results.append(result)
        return results
input_tensors = [torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32), torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=torch.float32), torch.tensor([[[3, 4]], [[5, 6]], [[7, 8]]], dtype=torch.float32), torch.tensor([[[2, 3, 4, 5], [6, 7, 8, 9]]], dtype=torch.float32), torch.tensor([[[10]], [[20]], [[30]], [[40]]], dtype=torch.float32)]
if __name__ == '__main__':
    module = PtModule()
    output = module.forward(input_tensors)
    for (i, out) in enumerate(output):
        print(f'Output {i}: {out}')