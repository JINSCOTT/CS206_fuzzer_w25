import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = 0
        for i in range(input_tensor.size(0)):
            result += torch.sum(input_tensor[i])  # Summing elements of each tensor
        return result

# Defining input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D
    torch.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]),  # 3D
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]]]),  # 3D
    torch.tensor([[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], [[7.5, 8.5, 9.5], [10.5, 11.5, 12.5]]]),  # 4D
    torch.tensor([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]])  # 4D
]

if __name__ == "__main__":
    module = PtModule()
    for tensor in input_tensors:
        output = module(tensor)
        print(f"Output for input tensor:\n{tensor}\nResult: {output.item()}\n")