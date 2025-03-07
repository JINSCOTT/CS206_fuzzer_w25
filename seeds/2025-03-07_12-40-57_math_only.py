import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example normal math operations
        x = x + 2          # Addition
        x = x - 1          # Subtraction
        x = x * 3          # Multiplication
        x = x / 4          # Division
        return x

# Defining input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),               # 3D tensor (1, 2, 2)
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),   # 3D tensor (1, 2, 3)
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),         # 4D tensor (2, 2, 1, 1)
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),    # 4D tensor (2, 2, 2, 1)
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])       # 2D tensor (2, 3)
]

if __name__ == "__main__":
    model = PtModule()
    for idx, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {idx + 1}:\n{output}")