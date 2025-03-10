import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example mathematical operations
        return (x + 2) * (x - 3) / (x + 1) ** 2

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),         # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),  # 4D tensor
    torch.tensor([[1, 2], [3, 4], [5, 6]]),            # 2D tensor with integers
    torch.tensor([3.0, 4.0])                            # 1D tensor
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output_tensor = model(input_tensor)
        print(f"Input: {input_tensor}, Output: {output_tensor}")