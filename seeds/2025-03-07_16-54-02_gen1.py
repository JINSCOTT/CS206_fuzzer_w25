import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensors):
        results = []
        for tensor in input_tensors:
            result = torch.mean(tensor)  # Example operation: mean of the tensor
            results.append(result)
        return results

# Defining input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),                  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),   # 3D tensor
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),     # 2D tensor with floats
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]]),   # 3D tensor with depth
    torch.tensor([[[[1, 2], [3, 4]]]])                     # 4D tensor
]

if __name__ == "__main__":
    module = PtModule()
    output = module(input_tensors)
    print(output)