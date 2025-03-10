import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = tensor.sum(dim=0) * 2  # Example operation: sum over the first dimension and multiply by 2
            results.append(result)
        return results

# Defining input tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
    torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]),
    torch.tensor([1, 2, 3, 4, 5]).view(1, 5),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]])
]

if __name__ == "__main__":
    pt_module = PtModule()
    outputs = pt_module(input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output {i+1}: {output}")