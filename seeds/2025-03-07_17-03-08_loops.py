import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Using basic math operators
        x = input_tensor + 2          # Addition
        y = x * 3                     # Multiplication
        z = y - 5                     # Subtraction
        result = z / 4                # Division
        
        # Looping through the dimensions of the tensor
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = result[i, j] ** 2  # Squaring each element

        return result

# Input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=torch.float32),
    torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32),
    torch.tensor([[[1]], [[2]], [[3]], [[4]]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4], [5, 6]]], dtype=torch.float32)
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)