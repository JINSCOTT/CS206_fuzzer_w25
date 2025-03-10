import torch

class PtModule:
    def __init__(self):
        pass

    def forward(self, input_tensor):
        # Example math operations
        result = input_tensor * 2  # simple multiplication
        for i in range(result.shape[0]):
            result[i] = result[i] + i  # Adding the index to each element
        return result

# Define input tensors with explicit values
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),
    torch.tensor([[[[1, 0], [0, 1]], [[1, 1], [0, 0]]], [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]], dtype=torch.float32),
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], dtype=torch.float32),
    torch.tensor([[[[1.5]], [[2.5]], [[3.5]]]], dtype=torch.float32)
]

if __name__ == "__main__":
    module = PtModule()
    for tensor in input_tensors:
        output = module.forward(tensor)
        print(output)