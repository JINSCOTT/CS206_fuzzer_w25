import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = input_tensor.clone()
        # Apply multiple math operations
        result = result + 5  # Addition
        result = result * 2  # Multiplication
        result = result - 3  # Subtraction

        # Loop over the tensor for some operations
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i][j] = torch.sqrt(result[i][j])  # Square root

        return result

input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[5, 10], [15, 20], [25, 30]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32),
    torch.tensor([[1, 4, 9], [16, 25, 36]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input_tensor[{i}]:\n{output}")