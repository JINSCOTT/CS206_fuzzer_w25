import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Performing some ordinary math operations
        return (x * 2) + 3 - 5 / 2

# Define 5 input tensors with explicit values
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),           # 2D tensor
    torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]), # 3D tensor
    torch.tensor([[[[1.0]], [[2.0]]]]),                   # 4D tensor
    torch.tensor([0.0, 1.0, 2.0, 3.0]),                   # 1D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]) # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")