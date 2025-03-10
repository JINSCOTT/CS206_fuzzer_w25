import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Perform some normal math operations
        x = x + 2       # Addition
        x = x * 3       # Multiplication
        x = x - 4       # Subtraction
        x = x / 5       # Division
        return x

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),             # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),           # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),        # 4D tensor
    torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]), # 2D tensor
    torch.tensor([[[5.0], [6.0]], [[7.0], [8.0]]])      # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")