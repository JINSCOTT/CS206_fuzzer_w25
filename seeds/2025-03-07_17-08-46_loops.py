import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        outputs = []
        for tensor in inputs:
            # Example of mathematical operations
            sum_tensor = torch.sum(tensor)
            mean_tensor = torch.mean(tensor)
            square_tensor = tensor ** 2
            outputs.append((sum_tensor, mean_tensor, square_tensor))
        return outputs

# Define input tensors with explicit values
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[5, 6, 7], [8, 9, 10], [11, 12, 13]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32)  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    results = model(input_tensors)
    for idx, result in enumerate(results):
        print(f"Output for tensor {idx}: Sum = {result[0]}, Mean = {result[1]}, Squared Tensor = {result[2]}")