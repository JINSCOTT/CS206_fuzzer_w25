import torch

class PtModule:
    def __init__(self):
        pass
    
    def forward(self, input_tensor):
        # Perform some mathematical operations
        output_tensor = input_tensor * 2  # Example operation: scaling the input
        for i in range(output_tensor.size(0)):  # Iterate through the first dimension
            output_tensor[i] = output_tensor[i] + 1  # Example operation: adding 1 to each tensor
        return output_tensor

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),  # 2D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),  # 4D tensor
    torch.tensor([[0, -1], [2, -3], [4, -5]]),  # 2D tensor
    torch.tensor([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]])  # 3D tensor
]

# Main section
if __name__ == "__main__":
    module = PtModule()
    for tensor in input_tensors:
        output = module.forward(tensor)
        print(f"Input Tensor:\n{tensor}\nOutput Tensor:\n{output}\n")