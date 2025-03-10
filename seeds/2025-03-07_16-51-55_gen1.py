import torch

class PtModule:
    def __init__(self):
        pass

    def forward(self, input_tensors):
        results = []
        for tensor in input_tensors:
            # Example operation: simple element-wise addition of a constant
            result = tensor + 5
            results.append(result)
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),                # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), # 3D tensor
    torch.tensor([1, 2, 3, 4]),                          # 1D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]]]),           # 4D tensor
    torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # 2D tensor
]

if __name__ == "__main__":
    module = PtModule()
    outputs = module.forward(input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output {i+1}:")
        print(output)