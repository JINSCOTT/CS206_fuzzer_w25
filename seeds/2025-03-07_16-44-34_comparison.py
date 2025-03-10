import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Math operations
        add_result = x + 5
        sub_result = x - 3
        mul_result = x * 2
        div_result = x / 2
        pow_result = x ** 2

        # Comparison operations
        greater_than = x > 2
        less_than = x < 5
        equal_to = x == 3

        return add_result, sub_result, mul_result, div_result, pow_result, greater_than, less_than, equal_to

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32),
    torch.tensor([[1], [2], [3]], dtype=torch.float32),
    torch.tensor([[[7, 8], [9, 10]], [[11, 12], [13, 14]]], dtype=torch.float32),
    torch.tensor([5, 4, 3, 2, 1], dtype=torch.float32),
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        outputs = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutputs:\n{outputs}\n")