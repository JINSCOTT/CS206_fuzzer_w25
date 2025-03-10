import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
    
    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            # Performing a simple mathematical operation
            result = input_tensor * 2 + 3
            results.append(result)
        return results

# Input tensors with explicit values
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),           # 2D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),     # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([1, 2, 3, 4], dtype=torch.float32)                 # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)