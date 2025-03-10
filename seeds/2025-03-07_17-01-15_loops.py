import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
    
    def forward(self, x):
        # Example of multiple math operations
        x = x + 10
        x = x * 2
        
        # Loop through the tensor dimensions
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                x[i, j] = x[i, j] ** 2  # Square elements
        
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),              # 2D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]), # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),    # 4D tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),              # Another 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]) # Another 3D tensor
]

# Main section to run the module
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print("Input Tensor:\n", input_tensor)
        print("Output Tensor:\n", output)