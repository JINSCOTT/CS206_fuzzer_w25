import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
        
    def forward(self, x):
        # Using multiple math operators
        x = x + 2
        x = x * 3
        x = x / 4
        
        # Applying a loop for summation
        for i in range(x.shape[1]):
            x[:, i] = x[:, i] + i
        
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[9, 10, 11], [12, 13, 14]]], dtype=torch.float32),
    torch.tensor([[[1], [2], [3], [4]]], dtype=torch.float32),
    torch.tensor([[[5, 6], [7, 8], [9, 10]]], dtype=torch.float32),
    torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")