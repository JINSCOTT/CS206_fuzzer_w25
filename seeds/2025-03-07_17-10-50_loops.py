import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Example of multiple math operations
        result = (input_tensor * 2) + 3

        # Looping through each element
        for i in range(result.size(0)):
            for j in range(result.size(1)):
                result[i, j] = result[i, j] / (i + 1)

        return result

# Input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D Tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),  # 3D Tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),  # 4D Tensor
    torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]),  # 4D Tensor
    torch.tensor([[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]]])  # 4D Tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)