import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = input_tensor * 2  # Simple operation: multiply input by 2
        for i in range(result.size(0)):
            result[i] += i  # Another operation: add index to each element
        return result

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),  # 4D tensor
    torch.tensor([[5, 10, 15], [20, 25, 30], [35, 40, 45]]),  # 2D tensor
    torch.tensor([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input Tensor:\n{tensor}\nOutput Tensor:\n{output}\n")