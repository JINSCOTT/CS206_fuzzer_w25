import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Assume some normal operations, such as addition, subtraction, multiplication, and division
        x = x + 2  # Add 2
        x = x - 1  # Subtract 1
        x = x * 3  # Multiply by 3
        x = x / 2  # Divide by 2
        return x

input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
    torch.tensor([[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]]]),
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    torch.tensor([[[[1.0]]]])
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)