import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Performing some normal math operations
        x = x + 2   # Addition
        x = x - 1   # Subtraction
        x = x * 3   # Multiplication
        x = x / 2   # Division
        return x

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),          # 2D tensor (2x2)
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),        # 3D tensor (1x2x2)
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),     # 4D tensor (1x1x2x2)
    torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),  # 2D tensor (3x2)
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])  # 4D tensor (1x2x2x1)
]

def main():
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")

if __name__ == "__main__":
    main()