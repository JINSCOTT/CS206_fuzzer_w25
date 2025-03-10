import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Using mathematical operators
        output_tensor = (input_tensor * 2) + 1
        
        # Looping through the dimensions of the tensor
        for i in range(output_tensor.size(0)):
            for j in range(output_tensor.size(1)):
                output_tensor[i, j] = output_tensor[i, j] ** 2
                
        return output_tensor

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[5, 10], [15, 20], [25, 30]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1]], [[2]], [[3]], [[4]]], dtype=torch.float32)  # 4D tensor
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print("Input tensor:\n", tensor)
        print("Output tensor:\n", output)