import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example of some normal math operations
        y = x + 2         # Addition
        y = y * 3         # Multiplication
        y = y - 5         # Subtraction
        y = y / 2         # Division
        return y

# Define the input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[5.0, 6.0]], [[7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[9.0, 10.0], [11.0, 12.0]]),  # 2D tensor
    torch.tensor([[[13.0], [14.0]]]),  # 3D tensor
    torch.tensor([[[[15.0]]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for idx, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {idx}: \n{output}")