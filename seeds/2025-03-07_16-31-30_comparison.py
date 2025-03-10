import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        added = x + 2
        subtracted = x - 3
        multiplied = x * 4
        divided = x / 5
        greater_than = x > 1
        less_than = x < 5
        return added, subtracted, multiplied, divided, greater_than, less_than

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 3D tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),  # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),  # 2D tensor
    torch.tensor([[[9.0], [10.0]], [[11.0], [12.0]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        results = model(input_tensor)
        print(f"Input:\n{input_tensor}\nResults:\n{results}\n")