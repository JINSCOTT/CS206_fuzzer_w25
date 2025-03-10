import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2, x3, x4, x5):
        # Mathematical operations
        addition = x1 + x2
        subtraction = x3 - x4
        multiplication = x2 * x5
        division = x4 / (x3 + 1e-5)  # Adding a small epsilon to prevent division by zero
        power = x1 ** 2

        # Comparison operations
        greater_than = x2 > x3
        less_than = x4 < x5
        equal_to = x1 == x2
        not_equal_to = x3 != x5
        logical_and = torch.logical_and(greater_than, less_than)

        return addition, subtraction, multiplication, division, power, greater_than, less_than, equal_to, not_equal_to, logical_and

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),                   # 2D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),     # 3D tensor
    torch.tensor([1, 2, 3, 4]),                               # 1D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),     # 4D tensor
    torch.tensor([[0.5, 1.5], [2.5, 3.5]])                   # 2D tensor
]

# Main section to check runnability
if __name__ == "__main__":
    model = PtModule()
    outputs = model(*input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output {i+1}: {output}")