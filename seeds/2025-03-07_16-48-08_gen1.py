import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Perform a simple operation: calculate the mean and add to each element
            mean_val = tensor.mean()
            result = tensor + mean_val
            results.append(result)
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]),  # 4D tensor
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor
    torch.tensor([[[0.0], [1.0]], [[2.0], [3.0]], [[4.0], [5.0]]]),  # 3D tensor
    torch.tensor([[[2.0, 3.0]], [[5.0, 6.0]], [[8.0, 9.0]], [[11.0, 12.0]]])  # 4D tensor
]

if __name__ == "__main__":
    pt_module = PtModule()
    outputs = pt_module(input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output {i+1}:\n{output}\n")