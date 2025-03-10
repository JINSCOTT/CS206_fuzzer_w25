import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Perform some operations with input_tensor
        result = input_tensor.clone()
        
        # Example of addition
        result = result + 2
        
        # Example of multiplication
        result = result * 3
        
        # Loop through the last dimension and apply a function
        for i in range(result.size(-1)):
            result[..., i] = result[..., i] - 1
        
        return result

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[1], [2], [3]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)   # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')