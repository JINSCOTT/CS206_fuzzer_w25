import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Example math operations: element-wise addition and multiplication
        result = input_tensor * 2
        
        # For loop to perform some operation on tensors
        for i in range(result.size(0)):
            result[i] = result[i] + 1
            
        return result

# Define input tensors with explicit values
input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]]),  # 4D tensor
    torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),  # 2D tensor
    torch.tensor([[[[3.0, 2.0]], [[1.0, 0.0]]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)