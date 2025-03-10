import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = {}
        result['addition'] = x + 2
        result['subtraction'] = x - 2
        result['multiplication'] = x * 2
        result['division'] = x / 2
        result['power'] = x ** 2
        result['greater_than'] = x > 1
        result['less_than'] = x < 1
        return result

# Inputs
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32),
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32),
    torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=torch.float32),
    torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        print(f"Input:\n{tensor}\nOutput:\n{model(tensor)}\n")