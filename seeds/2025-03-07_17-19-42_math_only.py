import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations: addition, subtraction, multiplication, and division
        x1 = x + 2
        x2 = x - 1
        x3 = x * 3
        x4 = x / 2
        return x1, x2, x3, x4

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),        # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]), # 4D tensor
    torch.tensor([[1, 2], [3, 4], [5, 6]]),          # 2D integer tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])          # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input: {input_tensor}\nOutput: {output}\n")