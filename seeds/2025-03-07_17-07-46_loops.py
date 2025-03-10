import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            # Apply some math operations
            added = input_tensor + 1
            multiplied = input_tensor * 2
            
            # Loop to apply max operation
            max_value = torch.max(input_tensor)
            results.append((added, multiplied, max_value))
        
        return results

# Sample input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),         # 2D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),      # 1D tensor
    torch.tensor([[[[1, 2]]]], dtype=torch.float32)               # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)