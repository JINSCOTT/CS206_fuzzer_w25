import torch

class PtModule:

    def __init__(self):
        pass

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            results.append(torch.pow(tensor, 2))
        return results
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=torch.float32), torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.float32), torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)]
if __name__ == '__main__':
    pt_module = PtModule()
    results = pt_module.forward(input_tensors)
    for (i, result) in enumerate(results):
        print(f'Output for tensor {torch.add(i, 1)}:\n{result}\n')