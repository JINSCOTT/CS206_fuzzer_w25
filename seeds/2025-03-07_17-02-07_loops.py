import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
    
    def forward(self, x):
        # Example operations
        result = x + 2  # Addition
        result = result * 3  # Multiplication
        
        # Applying a loop to process elements
        for i in range(result.size(0)):
            for j in range(result.size(1)):
                result[i, j] = result[i, j] - 1  # Subtraction in a loop
        
        return result

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),  # 2D
    torch.tensor([[[1, 2, 3]], [[4, 5, 6]]], dtype=torch.float32),  # 3D
    torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32),  # 4D
    torch.tensor([[5, 6], [7, 8]], dtype=torch.float32),  # 2D
    torch.tensor([[[[9]], [[10]]]], dtype=torch.float32)  # 4D
]

# Main section to check if it's runnable
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")