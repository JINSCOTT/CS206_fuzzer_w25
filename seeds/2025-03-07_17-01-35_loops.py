import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Example of using multiple math operators and loops
        output = input_tensor.clone()  # Clone the input tensor
        for i in range(output.size(0)):  # Loop over the first dimension
            for j in range(output.size(1)):  # Loop over the second dimension
                output[i, j] = output[i, j] * 2 + 5 - 3 / (1 + output[i, j])
        
        return output

# Input Tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]),  # 3D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),  # 4D tensor
    torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)