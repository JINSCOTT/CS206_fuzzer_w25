import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Perform a simple mathematical operation: element-wise multiplication by 2
            processed_tensor = tensor * 2
            results.append(processed_tensor)
        return results

# Inputs: 5 tensors with 3 to 4 dimensions
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]),  # 4D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])  # 4D tensor
]

if __name__ == "__main__":
    module = PtModule()
    output = module(input_tensors)
    for i, out in enumerate(output):
        print(f"Output tensor {i+1}:\n{out}\n")