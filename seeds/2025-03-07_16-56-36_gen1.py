import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Example operation: sum all elements in the tensor
            result = torch.sum(tensor)
            results.append(result)
        return results

# Input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),            # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([1, 2, 3, 4, 5]),                   # 1D tensor
    torch.tensor([[[[1, 2], [3, 4]]]]),              # 4D tensor
    torch.tensor([[10, 20, 30], [40, 50, 60]])      # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    results = model(input_tensors)
    for i, res in enumerate(results):
        print(f"Result for input tensor {i}: {res.item()}")