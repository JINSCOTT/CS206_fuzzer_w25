import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = tensor * 2 + 1  # Simple mathematical operation
            results.append(result)
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
    torch.tensor([[[13, 14], [15, 16]], [[17, 18], [19, 20]]]),
    torch.tensor([[[21], [22]], [[23], [24]]]),
    torch.tensor([[[25, 26, 27, 28]], [[29, 30, 31, 32]]]),
    torch.tensor([[[33, 34, 35], [36, 37, 38], [39, 40, 41]], [[42, 43, 44], [45, 46, 47], [48, 49, 50]]])
]

if __name__ == "__main__":
    module = PtModule()
    output = module(input_tensors)
    for o in output:
        print(o)