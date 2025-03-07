import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Applying some math operations and loops
        result = []
        for i in range(x.size(0)):
            temp = x[i] * 2  # Multiply each element by 2
            temp = temp + 1  # Add 1 to each element
            result.append(temp)
        return torch.stack(result)

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]]]),  # 4D tensor
    torch.tensor([[0, 1, 2], [3, 4, 5]]),  # 2D tensor
    torch.tensor([[[-1, -2], [-3, -4]], [[-5, -6], [-7, -8]]]),  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")