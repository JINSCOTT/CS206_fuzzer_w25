import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Performing some operations with the input tensor
        result = input_tensor * 2  # Multiplication
        result = result + 3        # Addition

        # Using a loop to modify each element
        for i in range(result.shape[0]):
            result[i] = result[i] - 1  # Subtraction

        return result

# Defining input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),    # 2D tensor
    torch.tensor([[[1.0, 2.0],[3.0, 4.0]], [[5.0, 6.0],[7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0],[3.0, 4.0]],[[5.0, 6.0],[7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([1, 2, 3, 4]),  # 1D tensor
    torch.tensor([[[10.0]]])     # 3D tensor with a single element
]

# Main section to test the module
if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input Tensor:\n{tensor}\nOutput Tensor:\n{output}\n")