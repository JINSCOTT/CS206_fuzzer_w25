import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Perform some mathematical operations and loops
        output = []
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                # Example operation: square the value and subtract 1
                val = x[i, j] ** 2 - 1
                output.append(val)
        return torch.tensor(output).view(x.size(0), x.size(1))

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]]),  # 4D tensor
    torch.tensor([[7.0, 8.0], [9.0, 10.0]]),  # 2D tensor
    torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]], [[[5.0, 6.0]], [[7.0, 8.0]]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for i, tensor in enumerate(input_tensors):
        print(f"Input Tensor {i+1}:\n{tensor}")
        output = model(tensor)
        print(f"Output after processing Tensor {i+1}:\n{output}\n")