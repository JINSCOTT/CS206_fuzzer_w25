import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        output = []
        for tensor in inputs:
            # Applying some mathematical operations
            tensor_sum = torch.sum(tensor)
            tensor_mean = torch.mean(tensor)
            tensor_max = torch.max(tensor)

            output.append((tensor_sum, tensor_mean, tensor_max))
        return output

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),  # 2D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),  # 4D tensor
    torch.tensor([[[-1, -2, -3], [-4, -5, -6]]]),  # 3D tensor with negative values
    torch.tensor([[0., 0.], [0., 0.]])  # 2D tensor of zeros
]

if __name__ == "__main__":
    model = PtModule()
    results = model(input_tensors)
    for result in results:
        print(result)