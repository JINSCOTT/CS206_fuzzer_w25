import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Mathematical operations
        add_result = x + 2
        mul_result = x * 3
        sub_result = x - 1
        div_result = x / 2
        
        # Comparison operations
        greater_than = x > 1
        less_than = x < 5
        equal_to = x == 2
        
        return add_result, mul_result, sub_result, div_result, greater_than, less_than, equal_to

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([1.0, 2.0, 3.0]),  # 1D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        result = model(tensor)
        print(result)