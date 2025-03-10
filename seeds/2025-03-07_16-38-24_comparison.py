import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Mathematical operations
        result1 = x + 5          # Addition
        result2 = x - 3          # Subtraction
        result3 = x * 2          # Multiplication
        result4 = x / 2          # Division
        result5 = x ** 2         # Exponentiation
        
        # Comparison operations
        result6 = x > 0          # Greater than
        result7 = x < 10         # Less than
        result8 = x == 5         # Equal to
        
        return result1, result2, result3, result4, result5, result6, result7, result8

input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),         # 2D tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),       # 3D tensor (actually treated as 2D because of shape)
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), # 3D tensor
    torch.tensor([1.0, 2.0, 3.0, 4.0]),                        # 1D tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Output for input tensor:\n{input_tensor}\n{output}\n")