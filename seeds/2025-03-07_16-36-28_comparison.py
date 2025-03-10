import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Multiple math operations
        sum_result = x + 2
        diff_result = x - 3
        prod_result = x * 4
        div_result = x / 5
        
        # Comparison operations
        greater_than = x > 1
        less_than_equal = x <= 3
        
        return sum_result, diff_result, prod_result, div_result, greater_than, less_than_equal

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
    torch.tensor([[1, 2, 3], [4, 5, 6]]),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]),
    torch.tensor([1.0, 2.0, 3.0])
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input tensor:\n{tensor}\nOutputs:\n{output}\n")