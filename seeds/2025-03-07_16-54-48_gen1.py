import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Performing some basic math operations for illustration
            tensor_sum = torch.sum(tensor)
            tensor_mean = torch.mean(tensor)
            results.append((tensor_sum, tensor_mean))
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]),  # 4D tensor
    torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]),  # 3D tensor
    torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])  # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)