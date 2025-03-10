import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Using various math operators
        result = x + 2
        result = result * 3
        result = result - 4
        result = result / 5

        # Using loops
        for i in range(3):
            result = result + i

        return result

# Defining input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32),
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]], dtype=torch.float32),
]

# Main section to check if script is runnable
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")