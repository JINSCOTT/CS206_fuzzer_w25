import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        added = x + 2
        subtracted = x - 2
        multiplied = x * 3
        divided = x / 4
        greater_than = x > 1
        less_than = x < 5
        
        # Return results as a tuple
        return added, subtracted, multiplied, divided, greater_than, less_than

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),
    torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]], dtype=torch.float32),
    torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.float32),
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print("Input:\n", input_tensor)
        print("Output:\n", output)