import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Example math operation: adding a constant and multiplying by a factor
            result = (tensor + 2) * 3
            results.append(result)
        return results

# Defining input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]),  # 4D tensor
    torch.tensor([[[10, 20, 30]], [[40, 50, 60]]]),  # 3D tensor
    torch.tensor([[1, 2], [3, 4], [5, 6]]),  # 2D tensor
    torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]])  # 3D tensor
]

if __name__ == "__main__":
    module = PtModule()
    output = module(input_tensors)
    for i, out in enumerate(output):
        print(f"Output of tensor {i}: \n{out}")