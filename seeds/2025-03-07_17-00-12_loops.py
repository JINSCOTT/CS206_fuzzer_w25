import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Example of math operations
            sum_tensor = torch.sum(tensor)
            mean_tensor = torch.mean(tensor)
            max_tensor = torch.max(tensor)
            min_tensor = torch.min(tensor)
            results.append((sum_tensor.item(), mean_tensor.item(), max_tensor.item(), min_tensor.item()))
        return results

# Define the input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),            # 2D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]), # 3D tensor
    torch.tensor([[[[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]]), # 4D tensor
    torch.tensor([[1, 2], [3, 4], [5, 6]]),              # Another 2D tensor
    torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)