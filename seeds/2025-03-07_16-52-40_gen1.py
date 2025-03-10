import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Performing a simple operation like adding 1 to each element
        output = []
        for i in range(x.size(0)):
            output.append(x[i] + 1)
        return torch.stack(output)

# Define the input tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
    torch.tensor([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]]),
    torch.tensor([[[0, 0], [0, 1]], [[1, 1], [1, 0]]]),
    torch.tensor([[[2, 3, 4, 5], [6, 7, 8, 9]]]),
    torch.tensor([[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]]])
]

if __name__ == "__main__":
    module = PtModule()
    for tensor in input_tensors:
        output = module(tensor)
        print(output)