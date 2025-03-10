import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Perform multiple math operations
        add_result = x + 5
        sub_result = x - 3
        mul_result = x * 2
        div_result = x / 4
        
        # Perform comparison operators
        gt_result = x > 2
        lt_result = x < 5
        
        return add_result, sub_result, mul_result, div_result, gt_result, lt_result

# Defining input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[1.0]]),  # 2D tensor with one element
    torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # 2D tensor with binary values
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")