import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Add
        add_result = x + 5
        
        # Subtract
        subtract_result = x - 3
        
        # Multiply
        multiply_result = x * 2
        
        # Divide
        divide_result = x / 2
        
        # Greater than
        greater_than_result = x > 2
        
        return add_result, subtract_result, multiply_result, divide_result, greater_than_result

input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),          # 2D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),  # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),  # 4D tensor
    torch.tensor([[10.0, 20.0], [30.0, 40.0]]),       # 2D tensor
    torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])      # 4D tensor
]

if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print("Input Tensor:\n", input_tensor)
        print("Outputs:\n", output)