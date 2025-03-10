import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Add, Subtract, Multiply, Divide
        add_result = x + 2
        sub_result = x - 2
        mul_result = x * 2
        div_result = x / 2
        
        # Comparisons
        greater_than = x > 1
        less_than = x < 3
        equal_to = x == 2
        
        return add_result, sub_result, mul_result, div_result, greater_than, less_than, equal_to

# Input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[1]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]]], dtype=torch.float32)  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for x in input_tensors:
        results = model(x)
        print(results)