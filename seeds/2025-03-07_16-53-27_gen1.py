import torch

class PtModule(torch.nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Perform some simple math operations for demonstration
            result = tensor * 2 + 3
            results.append(result)
        return results

# Input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]),  # 4D tensor
    torch.tensor([[7, 8, 9], [10, 11, 12]]),  # Another 2D tensor
    torch.tensor([[[10, 11], [12, 13]], [[14, 15], [16, 17]]])  # Another 3D tensor
]

if __name__ == "__main__":
    module = PtModule()
    output = module(input_tensors)
    for i, out in enumerate(output):
        print(f"Output for tensor {i + 1}:\n{out}")