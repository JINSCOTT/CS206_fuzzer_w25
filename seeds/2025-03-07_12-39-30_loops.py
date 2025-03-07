import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Applying basic math operations
            tensor_sum = torch.sum(tensor)
            tensor_mean = torch.mean(tensor)
            tensor_max = torch.max(tensor)
            tensor_min = torch.min(tensor)
            results.append((tensor_sum, tensor_mean, tensor_max, tensor_min))
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D tensor
    torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]),  # 3D tensor
    torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]),  # 3D tensor
    torch.tensor([[[1, 2, 3]], [[4, 5, 6]]]),  # 3D tensor
    torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)