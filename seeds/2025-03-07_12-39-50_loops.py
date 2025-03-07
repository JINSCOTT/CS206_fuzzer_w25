import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Multiply tensor by 2
            multiplied = tensor * 2
            
            # Add 1 to each element
            added = multiplied + 1
            
            # Calculate the mean
            mean_value = added.mean()
            
            results.append(mean_value)
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[10, 20], [30, 40], [50, 60]], dtype=torch.float32),
    torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32),
    torch.tensor([[[1]], [[2]], [[3]]], dtype=torch.float32),
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)