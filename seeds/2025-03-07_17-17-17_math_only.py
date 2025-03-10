import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations: addition, subtraction, multiplication, and division
        x1 = x + 2
        x2 = x - 3
        x3 = x * 4
        x4 = x / 5
        return x1, x2, x3, x4

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),         # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),  # 4D tensor
    torch.tensor([1.0, 2.0, 3.0]),                    # 1D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]) # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input Tensor:\n{tensor}\nOutput:\n{output}\n")