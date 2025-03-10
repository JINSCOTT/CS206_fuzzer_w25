import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Example math operations
        output_tensor = input_tensor + 2  # Addition
        output_tensor = output_tensor * 3  # Multiplication
        
        # Loop through the dimensions of the tensor
        for i in range(output_tensor.shape[0]):  # Loop over the first dimension
            for j in range(output_tensor.shape[1]):  # Loop over the second dimension
                output_tensor[i, j] = output_tensor[i, j] / 2  # Division
            
        return output_tensor

# Defining input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[9, 10], [11, 12]]),  # 2D tensor
    torch.tensor([[[13], [14]], [[15], [16]]]),  # 3D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]]])  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for inp in input_tensors:
        output = model(inp)
        print(f"Input:\n{inp}\nOutput:\n{output}\n")