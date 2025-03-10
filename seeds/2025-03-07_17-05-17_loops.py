import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            if input_tensor.dim() == 3:
                result = torch.sum(input_tensor, dim=1)  # Sum over the second dimension
            elif input_tensor.dim() == 4:
                result = torch.mean(input_tensor, dim=(1, 2))  # Mean over the spatial dimensions
            results.append(result)
        return results

# Input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]]),  # 4D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]]]),  # 3D tensor
    torch.tensor([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]]),  # 4D tensor
    torch.tensor([[[[5, 10, 15], [20, 25, 30]], [[35, 40, 45], [50, 55, 60]]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    for i, out in enumerate(output):
        print(f"Output for input {i}: \n{out}")