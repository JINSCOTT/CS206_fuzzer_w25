import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        add_result = x + 2
        sub_result = x - 3
        mul_result = x * 4
        div_result = x / 2
        greater_than_result = x > 5
        
        return add_result, sub_result, mul_result, div_result, greater_than_result

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32),
    torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32),
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
    torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        result = model(tensor)
        print(result)