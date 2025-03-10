import torch

class PtModule:
    def __init__(self):
        pass

    def forward(self, input_tensor):
        result = input_tensor.clone()  # Create a clone to hold the results
        for i in range(result.shape[0]):  # Iterate over the first dimension
            result[i] = result[i] * 2  # Simple operation: multiply by 2
        return result


input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # Shape (2, 2, 2)
    torch.tensor([[[9, 10, 11], [12, 13, 14]], [[15, 16, 17], [18, 19, 20]]]),  # Shape (2, 2, 3)
    torch.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]),  # Shape (2, 3, 2)
    torch.tensor([[[1, 2], [3, 4]]]),  # Shape (1, 2, 2)
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]])  # Shape (3, 2, 1)
]

if __name__ == "__main__":
    pt_module = PtModule()
    
    for input_tensor in input_tensors:
        output = pt_module.forward(input_tensor)
        print(f"Input: \n{input_tensor}\nOutput: \n{output}\n")