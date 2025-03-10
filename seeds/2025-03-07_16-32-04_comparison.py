import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        add_result = x + 2
        sub_result = x - 2
        mul_result = x * 2
        div_result = x / 2
        greater_than = x > 1
        less_than = x < 5
        
        return add_result, sub_result, mul_result, div_result, greater_than, less_than

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),           # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), # 4D tensor
    torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),  # 2D tensor
    torch.tensor([2.0]),                               # 1D tensor
]

# Main section
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)