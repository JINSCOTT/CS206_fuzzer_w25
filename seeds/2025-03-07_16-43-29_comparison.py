import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Mathematical operations
        add_result = x + 5
        sub_result = x - 3
        mul_result = x * 2
        div_result = x / 4

        # Comparison operations
        gt_result = x > 2
        lt_result = x < 5
        eq_result = x == 3

        return {
            'addition': add_result,
            'subtraction': sub_result,
            'multiplication': mul_result,
            'division': div_result,
            'greater_than': gt_result,
            'less_than': lt_result,
            'equal_to': eq_result
        }

# Input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[5, 10], [15, 20], [25, 30]], dtype=torch.float32),
    torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)