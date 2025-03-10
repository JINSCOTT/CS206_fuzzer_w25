import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            tensor = tensor + 2  # Addition
            tensor = tensor * 3  # Multiplication
            tensor = tensor / 2  # Division
            tensor = tensor - 1  # Subtraction
            results.append(tensor)
        return results

# Input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32),
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[[1, 2]], [[3, 4]]]], dtype=torch.float32),
    torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.float32)
]

# Main section
if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    for i, out in enumerate(output):
        print(f"Output for tensor {i}: {out}")