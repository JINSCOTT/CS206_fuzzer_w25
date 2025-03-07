import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Example operations: addition, multiplication, and loops
        result = input_tensor * 2  # Element-wise multiplication by 2
        result = result + 5         # Element-wise addition of 5
        
        # Loop through the tensor and apply a custom operation
        for i in range(result.size(0)):
            for j in range(result.size(1)):
                result[i, j] = result[i, j] - (i + j)  # Subtracting i + j from each element
        
        return result

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),  # 4D tensor
    torch.tensor([[10, 20], [30, 40]]),  # Another 2D tensor
    torch.tensor([[[[9],[10]], [[11],[12]]], [[[13],[14]], [[15],[16]]]])  # Another 4D tensor
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    module = PtModule()
    for tensor in input_tensors:
        output = module(tensor)
        print(f"Input Tensor:\n{tensor}\nOutput Tensor:\n{output}\n")