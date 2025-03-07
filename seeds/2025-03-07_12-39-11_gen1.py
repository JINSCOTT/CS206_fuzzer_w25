import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Perform some simple mathematical operations
            result = tensor * 2 + 1  # Example operation
            results.append(result)
        return results

# Defining input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    torch.tensor([[[1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0]]]),
    torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
]

if __name__ == "__main__":
    module = PtModule()
    output = module(input_tensors)
    for o in output:
        print(o)