import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example mathematical operations
        result = x + 2                      # Addition
        result = result * 3                 # Multiplication
        
        # Loop example
        for i in range(3):
            result = result - i              # Subtraction in loop
        
        return result

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),               # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),                # 4D tensor
    torch.tensor([[5, 6, 7], [8, 9, 10]]),                           # 2D integer tensor
    torch.tensor([[[[1], [2]], [[3], [4]], [[5], [6]]]])             # 4D integer tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor.float())
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")