import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            added = tensor + 2
            multiplied = added * 3
            results.append(multiplied)
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([1, 2, 3], dtype=torch.float32).view(1, 3),
    torch.tensor([[[0, 1, 2], [3, 4, 5]]], dtype=torch.float32),
    torch.tensor([[[[1], [2]], [[3], [4]]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)