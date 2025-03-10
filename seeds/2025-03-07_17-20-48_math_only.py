import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        return (x + 2) * 3 - 5 / 2

input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[5.0, 6.0, 7.0]]),            # 2D tensor
    torch.tensor([[[8.0]], [[9.0]], [[10.0]]]), # 3D tensor
    torch.tensor([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]), # 2D tensor
    torch.tensor([[[17.0, 18.0, 19.0]]])        # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input Tensor:\n{input_tensor}\nOutput Tensor:\n{output}\n")