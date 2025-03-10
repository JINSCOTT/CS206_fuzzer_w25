import torch

class PtModule:
    def __init__(self):
        pass

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Example operation: element-wise square
            result = tensor * tensor
            results.append(result)
        return results

# Define the input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),  # 3D tensor
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor
    torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]]),  # 4D tensor
    torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])  # 2D tensor
]

if __name__ == "__main__":
    module = PtModule()
    results = module.forward(input_tensors)
    for i, result in enumerate(results):
        print(f"Result for input tensor {i}: \n{result}\n")