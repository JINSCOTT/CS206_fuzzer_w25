import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Mathematical operations
        add_result = x + 2
        sub_result = x - 2
        mul_result = x * 3
        div_result = x / 2
        
        # Comparison operations
        greater_than = x > 1
        less_than = x < 5
        
        return add_result, sub_result, mul_result, div_result, greater_than, less_than

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),  # 4D tensor
    torch.tensor([[7, 8], [9, 10]]),  # 2D tensor with integers
    torch.tensor([[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]])  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        outputs = model(tensor)
        print(f"Input Tensor:\n{tensor}\nOutputs:\n{outputs}\n")