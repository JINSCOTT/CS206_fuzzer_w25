import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Applying multiple math operations
        addition = x + 5
        subtraction = x - 3
        multiplication = x * 2
        division = x / 4
        comparison = x > 0

        return {
            'addition': addition,
            'subtraction': subtraction,
            'multiplication': multiplication,
            'division': division,
            'comparison': comparison
        }

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32),
    torch.tensor([1, 2, 3], dtype=torch.float32),
    torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32),
    torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input Tensor: {input_tensor}")
        print("Output:", output)