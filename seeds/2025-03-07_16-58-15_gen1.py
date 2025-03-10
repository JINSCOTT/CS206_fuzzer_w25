import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output = torch.zeros_like(input_tensor)
        # Example operation: add 2 to each element and multiply by 3
        for i in range(input_tensor.size(0)):
            output[i] = (input_tensor[i] + 2) * 3
        return output

# Declare input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),
    torch.tensor([[[10, 20], [30, 40]], [[50, 60], [70, 80]], [[90, 100], [110, 120]]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]]], dtype=torch.float32),
    torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]], [[9, 10, 11, 12]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output_tensor = model(input_tensor)
        print(output_tensor)