import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations and comparisons
        added = x + 2
        subtracted = x - 3
        multiplied = x * 4
        divided = x / 5
        comparison = x > 1
        
        return added, subtracted, multiplied, divided, comparison

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),           # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32),     # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32)  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print("Output for tensor:\n", tensor)
        print("Added:\n", output[0])
        print("Subtracted:\n", output[1])
        print("Multiplied:\n", output[2])
        print("Divided:\n", output[3])
        print("Comparison:\n", output[4])