import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example of using multiple math operators: addition, multiplication, and mean calculation
        y = x + 2
        y = y * 3
        
        # Example of a loop: summing over the last dimension
        for i in range(y.shape[0]):  # Looping over the first dimension
            y[i] = y[i].sum(dim=-1)  # Summing the last dimension

        # Returning the modified tensor
        return y

# Defining input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),                      # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[10, 20, 30], [40, 50, 60]]),                   # 2D tensor
    torch.tensor([[9.0], [10.0], [11.0], [12.0]])               # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Output for input tensor {input_tensor}:\n{output}\n")