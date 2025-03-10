import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Sample operations
        return (x + 2) * 3 - 5 / 2

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0]]]]),  # 4D tensor
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor
    torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for idx, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {idx}: {output}")