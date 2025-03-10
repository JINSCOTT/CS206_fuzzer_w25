import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        a, b = inputs

        # Math operations
        addition = a + b
        subtraction = a - b
        multiplication = a * b
        division = a / (b + 1e-5)  # Avoid division by zero

        # Comparison operations
        greater_than = a > b
        less_than = a < b
        equal_to = a == b

        return addition, subtraction, multiplication, division, greater_than, less_than, equal_to

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]], dtype=torch.float32)   # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output {i+1}:\n{output}")