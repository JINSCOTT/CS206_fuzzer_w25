import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            # Applying some mathematical operations
            mean_val = input_tensor.mean()
            sum_val = input_tensor.sum()
            max_val = input_tensor.max()
            min_val = input_tensor.min()
            results.append((mean_val, sum_val, max_val, min_val))
        return results

# Defining input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),   # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[[1], [2]], [[3], [4]]]], [[[[5], [6]], [[7], [8]]]]]),  # 4D tensor
    torch.tensor([[9, 10, 11, 12], [13, 14, 15, 16]]),  # Another 2D tensor
    torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5])              # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)