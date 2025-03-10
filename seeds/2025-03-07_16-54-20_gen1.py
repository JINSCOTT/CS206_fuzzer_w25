import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensors):
        results = []
        for tensor in input_tensors:
            # Example operation: sum and multiply by 2
            result = torch.sum(tensor) * 2
            results.append(result)
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    torch.tensor([[9, 10, 11], [12, 13, 14]]),
    torch.tensor([[[15, 16], [17, 18]], [[19, 20], [21, 22]], [[23, 24], [25, 26]]]),
    torch.tensor([1.0, 2.0, 3.0]),
    torch.tensor([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]])
]

if __name__ == "__main__":
    model = PtModule()
    results = model(input_tensors)
    for i, result in enumerate(results):
        print(f"Result for input tensor {i}: {result.item()}")