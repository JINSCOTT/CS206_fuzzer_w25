import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
    
    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Performing some basic math operations
            added = tensor + 2
            multiplied = tensor * 3
            results.append((added, multiplied))
        
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[10, 20, 30]], dtype=torch.float32),  # 2D tensor with a single row
    torch.tensor([[[[0.1]]]], dtype=torch.float32)  # 4D tensor with a single value
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)