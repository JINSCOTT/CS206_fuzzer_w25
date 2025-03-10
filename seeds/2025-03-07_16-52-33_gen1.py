import torch

class PtModule:
    def __init__(self):
        pass
    
    def perform_operations(self, input_tensor):
        # Example mathematical operations
        result = input_tensor * 2  # Simple multiplication
        for i in range(result.shape[0]):  # For loop over the first dimension
            result[i] += torch.sum(input_tensor[i])  # Add sum of that slice to itself
        return result

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]]),  # 4D tensor
    torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], [[[7.0, 8.0, 9.0]], [[10.0, 11.0, 12.0]]], [[[13.0, 14.0, 15.0]], [[16.0, 17.0, 18.0]]]])  # 4D tensor
]

if __name__ == "__main__":
    module = PtModule()
    for input_tensor in input_tensors:
        output = module.perform_operations(input_tensor)
        print(f"Input Tensor:\n{input_tensor}\nOutput Tensor:\n{output}\n")