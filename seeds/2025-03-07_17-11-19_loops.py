import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example of addition
        x += 1

        # Example of multiplication
        x *= 2

        # Example of loop with subtraction
        for i in range(3):
            x -= i

        # Example of division
        x /= 2

        return x

# Input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]]),  # 3D tensor
    torch.tensor([[[[15.0, 16.0], [17.0, 18.0]], [[19.0, 20.0], [21.0, 22.0]]]]),  # 4D tensor
    torch.tensor([[[23.0], [24.0]], [[25.0], [26.0]]])  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)