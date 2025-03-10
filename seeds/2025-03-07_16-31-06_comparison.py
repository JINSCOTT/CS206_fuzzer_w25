import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Applying various math operations
        add_result = x + 5
        sub_result = x - 2
        mul_result = x * 3
        div_result = x / 2
        pow_result = x ** 2
        
        # Applying comparison operators
        greater_than = x > 3
        less_than = x < 4
        equal_to = x == 2
        
        return add_result, sub_result, mul_result, div_result, pow_result, greater_than, less_than, equal_to

# Define input_tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]),  # 4D tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # Another 2D tensor
    torch.tensor([[2.0, 4.0], [6.0, 8.0]]),  # Another 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)