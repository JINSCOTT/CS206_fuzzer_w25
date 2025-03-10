import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensors):
        results = []
        for tensor in input_tensors:
            tensor = tensor + 2  # Addition
            tensor = tensor * 3  # Multiplication
            tensor = tensor - 1  # Subtraction
            results.append(tensor)
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], dtype=torch.float32),
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32),
    torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)