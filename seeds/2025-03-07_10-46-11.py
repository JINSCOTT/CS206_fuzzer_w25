import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Using various operators and a loop
        result = []
        for i in range(x.size(0)):  # Loop through the first dimension
            temp = x[i] * 2 + 3  # Example operation: multiply by 2 and then add 3
            result.append(temp)
        return torch.stack(result)

# Input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]]]),  # 4D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]),  # 4D tensor
    torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])  # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input: {input_tensor}\nOutput: {output}\n")