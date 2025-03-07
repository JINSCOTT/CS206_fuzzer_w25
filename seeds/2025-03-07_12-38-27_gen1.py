import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = []
        for tensor in x:
            # Perform a simple operation (e.g., adding 1 to each element)
            result.append(tensor + 1)
        return result

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]),  # 4D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6]]),  # 2D tensor (will be handled)
    torch.tensor([[[1], [2]], [[3], [4]]]),  # 3D tensor with single element
    torch.tensor([10, 20, 30])  # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output {i}: {output}")