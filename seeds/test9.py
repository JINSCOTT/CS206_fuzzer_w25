import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example of using math operators
        result = x + 2  # Add 2 to each element
        result = result * 3  # Multiply by 3
        # Example of using loops
        for i in range(result.shape[1]):
            result[:, i] = result[:, i] ** 2  # Square each element in the second dimension
        return result

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2]]], dtype=torch.float32),
    torch.tensor([[[0, 1], [0, 1]], [[0, 1], [0, 1]]], dtype=torch.float32),
    torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32),
    torch.tensor([[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], dtype=torch.float32)
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input:\n{tensor}\nOutput:\n{output}\n")