import torch
import torch.nn as nn
import torch.nn.functional as F

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor1, input_tensor2):
        # Math operations
        addition = input_tensor1 + input_tensor2
        subtraction = input_tensor1 - input_tensor2
        multiplication = input_tensor1 * input_tensor2
        division = input_tensor1 / (input_tensor2 + 1e-6)  # Avoid division by zero
        
        # Comparison operations
        greater_than = input_tensor1 > input_tensor2
        less_than = input_tensor1 < input_tensor2
        equal_to = input_tensor1 == input_tensor2
        
        return addition, subtraction, multiplication, division, greater_than, less_than, equal_to

input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32)   # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for i in range(0, len(input_tensors), 2):
        output = model(input_tensors[i], input_tensors[i+1])
        print(f"Inputs:\n{input_tensors[i]}\n{input_tensors[i+1]}\nOutputs:\n{output}\n")