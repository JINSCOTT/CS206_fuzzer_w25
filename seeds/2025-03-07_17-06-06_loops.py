import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations using math operators and loops
        for i in range(x.size(0)):
            x[i] = x[i] * 2 + 1  # Multiply by 2 and add 1
        return x

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),  # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),  # Another 2D tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])   # Another 3D tensor
]

def main():
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)

if __name__ == "__main__":
    main()