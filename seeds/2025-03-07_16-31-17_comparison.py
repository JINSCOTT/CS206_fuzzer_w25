import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2, x3, x4, x5):
        # Math operations
        addition = x1 + x2
        subtraction = x3 - x4
        multiplication = x1 * x2
        division = x5 / (x2 + 1e-8)  # Add small value to avoid division by zero

        # Comparisons
        greater_than = x1 > x3
        less_than = x4 < x5
        equal_to = x1 == x2

        return addition, subtraction, multiplication, division, greater_than, less_than, equal_to

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),         # 2D tensor
    torch.tensor([[5, 6], [7, 8]]),         # 2D tensor
    torch.tensor([[[1, 2], [3, 4]]]),       # 3D tensor
    torch.tensor([[2, 3], [4, 5]]),         # 2D tensor
    torch.tensor([[[10], [20], [30]]])      # 3D tensor
]

# Main section to check if the code runs
if __name__ == "__main__":
    model = PtModule()
    results = model(*input_tensors)
    for res in results:
        print(res)