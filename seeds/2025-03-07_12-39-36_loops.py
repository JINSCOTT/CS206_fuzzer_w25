import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Example of various math operations
        output = input_tensor * 2          # Multiplication
        output = output + 5                 # Addition
        
        for i in range(2):                  # Loop
            output = output - (i + 1)       # Subtraction in a loop
            
        output = output / 3                 # Division
        return output

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),            # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([1, 2, 3]),                             # 1D tensor
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {i}: {output}")