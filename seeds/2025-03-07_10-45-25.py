import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Perform some operations using math operators and loops
        result = []
        for i in range(x.size(0)):  # Loop over the first dimension
            temp = x[i] * 2  # Multiply each element by 2
            temp = temp + 3  # Add 3 to each element
            temp = temp / 2  # Divide each element by 2
            result.append(temp)
        return torch.stack(result)

# Input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[9, 10, 11], [12, 13, 14]], dtype=torch.float32),
    torch.tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=torch.float32),
    torch.tensor([[15], [16], [17]], dtype=torch.float32),
    torch.tensor([[[18]], [[19]], [[20]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")