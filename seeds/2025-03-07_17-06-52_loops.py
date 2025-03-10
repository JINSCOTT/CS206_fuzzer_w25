import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Adding 1 to each element
        x = x + 1
        
        # Multiplying by 2
        x = x * 2

        # Loop to subtract 3 from each element in the tensor
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                x[i][j] = x[i][j] - 3

        return x

# Defining input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),                   # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),  # 4D tensor
    torch.tensor([[5, 10], [15, 20], [25, 30]]),      # 2D tensor
    torch.tensor([[[[1, 0, 2], [4, 2, 8]]]])          # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")