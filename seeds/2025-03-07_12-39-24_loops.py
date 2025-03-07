import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example of multiple math operations
        x1 = x + 2
        x2 = x * 3
        x3 = x - 1
        x4 = x / 2

        # Using loops to create a tensor
        result = torch.zeros_like(x)
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                result[i, j] = x1[i, j] + x2[i, j] - x3[i, j] * x4[i, j]

        return result

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),  # 2D tensor
    torch.tensor([[[[5.0, 6.0]], [[7.0, 8.0]], [[9.0, 10.0]]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input: {input_tensor}\nOutput: {output}\n")