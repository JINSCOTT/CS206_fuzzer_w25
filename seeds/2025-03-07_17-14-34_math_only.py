import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Perform some normal math operations
        x = x + 2        # Addition
        x = x - 1        # Subtraction
        x = x * 3        # Multiplication
        x = x / 4        # Division
        x = x ** 2       # Exponentiation
        return x

input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),
    torch.tensor([[1.0], [2.0], [3.0]]),
    torch.tensor([1.0, 2.0, 3.0, 4.0])
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input_tensor {i}: {output}")