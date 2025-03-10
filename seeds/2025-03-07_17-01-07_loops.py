import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example: Perform element-wise operations and loops
        result = torch.zeros_like(x)
        for i in range(x.size(0)):  # Loop over the first dimension
            for j in range(x.size(1)):  # Loop over the second dimension
                result[i, j] = x[i, j] * 2 + 5  # Example operation: multiply by 2 and add 5
        return result

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),   # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), # 3D tensor 
    torch.tensor([[1.0], [2.0], [3.0]]),       # 2D tensor with one column
    torch.tensor([1.0, 2.0, 3.0, 4.0]),        # 1D tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]) # 3D tensor
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input:\n{tensor}\nOutput:\n{output}\n")