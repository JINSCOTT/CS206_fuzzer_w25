import torch

class PtModule(torch.nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Simple mathematical operation: element-wise square
            result = tensor ** 2
            results.append(result)
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=torch.float32),
    torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], dtype=torch.float32),
    torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]], dtype=torch.float32),
    torch.tensor([[[2, 3], [4, 5], [6, 7]]], dtype=torch.float32),
]

if __name__ == "__main__":
    # Create a module instance
    pt_module = PtModule()
    
    # Run the module with the input tensors
    outputs = pt_module(input_tensors)
    
    # Print the results
    for i, output in enumerate(outputs):
        print(f"Output of tensor {i+1}:")
        print(output)