import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Apply multiple math operations and comparisons
        results = {}
        results['addition'] = x + 2
        results['subtraction'] = x - 2
        results['multiplication'] = x * 2
        results['division'] = x / 2
        results['greater_than'] = x > 3
        results['less_than'] = x < 3
        results['equal_to'] = x == 2
        return results

input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
    torch.tensor([[5, 6, 7], [8, 9, 10]]),
    torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]),
    torch.tensor([[2, 2], [2, 2]])
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)