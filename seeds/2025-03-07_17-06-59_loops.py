import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Example: Performing basic math operations
        output = input_tensor * 2  # Multiply by 2
        output = output + 3        # Add 3
        output = output - 1        # Subtract 1
        output = output / 4        # Divide by 4
        
        # Example loop
        for i in range(5):
            output = output + i  # Add the loop index
            
        return output

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),                # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), # 3D tensor
    torch.tensor([[1], [2], [3], [4]]),                  # 2D tensor with single column
    torch.tensor([[[1]], [[2]], [[3]], [[4]]]),         # 4D tensor
    torch.tensor([1, 2, 3, 4])                           # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for i, tensor in enumerate(input_tensors):
        output = model(tensor)
        print(f"Output for input tensor {i}:\n{output}")