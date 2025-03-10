import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Multiple mathematical operations
        addition = x + 2
        subtraction = x - 3
        multiplication = x * 4
        division = x / 5
        
        # Comparisons
        greater_than = x > 1
        less_than = x < 2
        equal_to = x == 2
        
        return (addition, subtraction, multiplication, division, greater_than, less_than, equal_to)

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[1], [2], [3], [4]], dtype=torch.float32),
    torch.tensor([[[[1], [2]], [[3], [4]]]], dtype=torch.float32)
]

# Main section to check if module is runnable
if __name__ == "__main__":
    model = PtModule()
    for inp in input_tensors:
        result = model(inp)
        print(f"Input:\n{inp}\nResults:\n{result}\n")