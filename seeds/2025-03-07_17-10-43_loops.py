import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = input_tensor

        # Example of mathematical operations
        result = result + 5  # Addition
        result = result * 2  # Multiplication
        result = result - 3  # Subtraction
        
        # Loop example
        for i in range(5):
            result = result + i

        return result

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]]),              # 4D tensor
    torch.tensor([[[9.0, 10.0], [11.0, 12.0]]] ),                      # 3D tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),                             # 2D tensor
    torch.tensor([1.0])                                                 # 1D tensor
]

# Main section
if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {i+1}:\n{output}") 