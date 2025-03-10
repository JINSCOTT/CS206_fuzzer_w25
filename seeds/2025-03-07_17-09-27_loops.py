import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Example math operations
            tensor_sum = torch.sum(tensor)
            tensor_mean = torch.mean(tensor)
            tensor_max = torch.max(tensor)
            results.append((tensor_sum, tensor_mean, tensor_max))
        return results

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),   # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[9, 10, 11], [12, 13, 14]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[[15]], [[16]]], [[[17]], [[18]]]], dtype=torch.float32)  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)