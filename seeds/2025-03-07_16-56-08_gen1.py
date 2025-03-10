import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Simple math operation: sum of all elements
            results.append(torch.sum(tensor))
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 4D tensor
    torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]),  # 4D tensor
    torch.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]),  # 4D tensor
    torch.tensor([[10, 20, 30], [40, 50, 60]]),  # 3D tensor
    torch.tensor([[1], [2], [3], [4]]),  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)