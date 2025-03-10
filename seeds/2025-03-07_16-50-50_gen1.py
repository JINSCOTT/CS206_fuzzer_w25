import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Perform some mathematical operations
        output = input_tensor * 2  # Scale the tensor by 2
        for i in range(output.shape[0]):
            output[i] = output[i] + i  # Add the loop index to each slice
        return output

# Input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),                   # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),     # 3D tensor
    torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]),  # 4D tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),                 # Another 2D tensor
    torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]])           # 3D tensor with a different shape
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output_tensor = model(tensor)
        print(f"Input Tensor:\n{tensor}\nOutput Tensor:\n{output_tensor}\n")