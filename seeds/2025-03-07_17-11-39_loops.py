import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Initialize a result tensor with zeros
        result = torch.zeros_like(x)

        # Loop over the dimensions of the input tensor
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    # Apply a mathematical operation
                    result[i, j, k] = x[i, j, k] * 2 + 3 - 1

        return result

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[1, 2], [3, 4], [5, 6]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]]], dtype=torch.float32)  # 4D tensor
]

# Main section to test if the module is runnable
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")