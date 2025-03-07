import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example of various math operations
        add_result = x + 5
        sub_result = x - 2
        mul_result = x * 3
        div_result = x / 2

        # Example of comparisons
        gt_result = x > 0
        lt_result = x < 10
        eq_result = x == 5
        
        return add_result, sub_result, mul_result, div_result, gt_result, lt_result, eq_result

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),                    # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]]]),            # 3D tensor
    torch.tensor([[[[1, 2, 3], [4, 5, 6]]]]),         # 4D tensor
    torch.tensor([[0.5, 1.5], [2.5, 3.5]]),            # 2D tensor (float)
    torch.tensor([[[1], [2], [3]], [[4], [5], [6]]])  # 3D tensor (int)
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")