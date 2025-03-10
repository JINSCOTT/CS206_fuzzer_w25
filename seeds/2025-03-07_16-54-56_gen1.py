import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        results = []
        for i in range(x.shape[0]):
            result = x[i] + 5  # simple addition operation
            results.append(result)
        return torch.stack(results)

# Input tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
    torch.tensor([[[13, 14], [15, 16]], [[17, 18], [19, 20]]]),
    torch.tensor([[21, 22, 23], [24, 25, 26]]),
    torch.tensor([[[27], [28]], [[29], [30]], [[31], [32]]]),
    torch.tensor([[[33, 34, 35, 36]], [[37, 38, 39, 40]]])
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)