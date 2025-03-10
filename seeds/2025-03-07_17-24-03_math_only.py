import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        return (x + 2) * 3 - 5 / 2

# Defining the input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # Shape (1, 2, 2)
    torch.tensor([[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]]),  # Shape (1, 2, 3)
    torch.tensor([[[11.0]], [[12.0]], [[13.0]]]),  # Shape (3, 1, 1)
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),  # Shape (1, 2, 2, 1)
    torch.tensor([[[[1.0, 2.0]]]])  # Shape (1, 1, 1, 2)
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output {i + 1}:\n{output}")