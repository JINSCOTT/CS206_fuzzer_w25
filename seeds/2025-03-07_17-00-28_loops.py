import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Using multiple math operators and loops
        result = []
        for i in range(x.size(0)):  # Loop through the first dimension
            batch_result = []
            for j in range(x.size(1)):  # Loop through the second dimension
                # Example operations
                value = x[i, j]
                squared = value ** 2
                doubled = value * 2
                summed = squared + doubled
                batch_result.append(summed)
            result.append(batch_result)
        
        return torch.tensor(result)

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),  # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),  # another 2D tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])   # another 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input:\n{tensor}\nOutput:\n{output}\n")