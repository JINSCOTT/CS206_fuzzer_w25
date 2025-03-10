import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations: addition, subtraction, multiplication, and division
        x1 = x + 5
        x2 = x - 3
        x3 = x * 2
        x4 = x / (torch.tensor(1.0) + x)  # avoid division by zero
        return x1, x2, x3, x4

# Defining input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([1, 2, 3, 4], dtype=torch.float32)  # 1D tensor
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        outputs = model(tensor)
        print(f"Input Tensor:\n{tensor}\nOutputs:\n{outputs}\n")