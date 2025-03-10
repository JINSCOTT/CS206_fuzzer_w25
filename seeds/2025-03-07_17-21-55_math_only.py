import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Perform some normal math operations
        x = x + 2       # Addition
        x = x - 1       # Subtraction
        x = x * 3       # Multiplication
        x = x / 2       # Division
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),          # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),       # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),            # 2D tensor
    torch.tensor([[[9.0]]]),                           # 3D tensor with single value
    torch.tensor([[[[10.0]]]])                         # 4D tensor with single value
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    
    for i, tensor in enumerate(input_tensors):
        output = model(tensor)
        print(f"Output for input tensor {i+1}:\n{output}\n")