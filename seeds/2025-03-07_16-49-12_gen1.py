import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Performing some basic mathematical operations
        result = input_tensor * 2  # Example operation: multiply by 2
        for i in range(result.shape[0]):
            result[i] = result[i] + 1  # Example operation: adding 1 to each tensor
        return result

# Input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D Tensor
    torch.tensor([[1, 2, 3], [4, 5, 6]]),                # 2D Tensor
    torch.tensor([[1, 2], [3, 4], [5, 6]]),              # 2D Tensor
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]]),  # 3D Tensor
    torch.tensor([[[[1, 2], [3, 4]]]])                    # 4D Tensor
]

if __name__ == "__main__":
    pt_module = PtModule()
    for tensor in input_tensors:
        output = pt_module(tensor)
        print("Input Tensor:\n", tensor)
        print("Output Tensor:\n", output)