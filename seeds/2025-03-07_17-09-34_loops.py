import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Example operations
        output_tensor = input_tensor * 2  # Multiplication
        output_tensor = output_tensor + 3  # Addition
        
        for i in range(output_tensor.size(0)):
            output_tensor[i] = output_tensor[i] - 1  # Subtraction
        
        output_tensor = torch.pow(output_tensor, 2)  # Exponentiation
        return output_tensor

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),              # 2D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]]]),   # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),      # 2D tensor with float values
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")