import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_data):
        results = []
        for tensor in input_data:
            # Simple math operation: element-wise square and then sum
            result = torch.sum(tensor ** 2)
            results.append(result)
        return results

# Defining input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),          # 2D Tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),  # 3D Tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),  # 2D Tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]),  # 3D Tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])           # 4D Tensor
]

if __name__ == "__main__":
    pt_module = PtModule()
    outputs = pt_module(input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output for tensor {i+1}: {output.item()}")