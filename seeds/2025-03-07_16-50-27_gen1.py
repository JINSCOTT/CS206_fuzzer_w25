import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            # Perform some operations, e.g., sum and mean
            tensor_sum = input_tensor.sum()
            tensor_mean = input_tensor.mean()
            results.append((tensor_sum, tensor_mean))
        return results

# Defining input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]], [[[5.0], [6.0]], [[7.0], [8.0]]]]),  # 4D tensor
    torch.tensor([[[-1.0, -2.0], [-3.0, -4.0]], [[-5.0, -6.0], [-7.0, -8.0]]]),  # 3D tensor
    torch.tensor([[[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)