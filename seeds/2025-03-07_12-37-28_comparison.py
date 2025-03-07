import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Add
        add_result = x + 2
        
        # Subtract
        sub_result = x - 1
        
        # Multiply
        mul_result = x * 3
        
        # Divide
        div_result = x / 4
        
        # Comparisons
        greater_than_result = x > 1
        less_than_result = x < 5
        
        return add_result, sub_result, mul_result, div_result, greater_than_result, less_than_result

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),   # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), # 3D tensor
    torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]), # 4D tensor
    torch.tensor([[5.0], [6.0]]),              # 2D tensor
    torch.tensor([10.0])                        # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input tensor:\n{input_tensor}\nOutputs:\n{output}\n")