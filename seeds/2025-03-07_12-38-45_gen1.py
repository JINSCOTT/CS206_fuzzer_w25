import torch

class PtModule:
    def __init__(self):
        pass

    def forward(self, input_tensor):
        results = []
        for i in range(input_tensor.size(0)):
            result = input_tensor[i] * 2  # Example operation: Multiply by 2
            results.append(result)
        return torch.stack(results)

# Define the input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[1, 1, 1], [2, 2, 2]]], dtype=torch.float32),
    torch.tensor([[[0, 0], [1, 1]], [[2, 2], [3, 3]], [[4, 4], [5, 5]]], dtype=torch.float32),
    torch.tensor([[[1, 0], [0, 1], [1, 1]]], dtype=torch.float32),
    torch.tensor([[[10]], [[20]], [[30]], [[40]]], dtype=torch.float32)
]

if __name__ == "__main__":
    module = PtModule()
    for tensor in input_tensors:
        output = module.forward(tensor)
        print(output)