import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example mathematical operations and loops
        result = []
        for i in range(x.size(0)):
            temp = x[i] * 2  # Multiply by 2
            temp = temp + 3  # Add 3
            temp = temp / 4  # Divide by 4
            result.append(temp)
        return torch.stack(result)

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[2, 4], [6, 8], [10, 12]]], dtype=torch.float32),
    torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=torch.float32),
    torch.tensor([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]], dtype=torch.float32),
    torch.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')