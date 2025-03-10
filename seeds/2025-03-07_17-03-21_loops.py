import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            sum_tensor = torch.sum(input_tensor)
            mean_tensor = torch.mean(input_tensor)
            std_tensor = torch.std(input_tensor)
            results.append((sum_tensor, mean_tensor, std_tensor))
        return results

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),               # 2D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32),   # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32)        # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)