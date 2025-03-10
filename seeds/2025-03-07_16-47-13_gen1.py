import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            result = torch.sum(input_tensor)  # Simple operation: sum of elements
            results.append(result)
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D Tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D Tensor
    torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]]),      # 4D Tensor
    torch.tensor([[10, 20, 30], [40, 50, 60]]),          # 2D Tensor
    torch.tensor([[[[10]], [[20]], [[30]]], [[[40]], [[50]], [[60]]]])  # 4D Tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)