import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Example math operations
        output = input_tensor + 2  # Addition
        output = output * 3         # Multiplication
        output = output - 1         # Subtraction
        output = output / 2         # Division

        # Looping over the tensor
        for i in range(output.size(0)):
            output[i] = output[i] ** 2  # Squaring each element

        return output

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),         # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[1.5, 2.5], [3.5, 4.5]]),       # 2D tensor with floats
    torch.tensor([[[1], [2]], [[3], [4]]]),       # 3D tensor with single column
    torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]]) # 4D tensor
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")