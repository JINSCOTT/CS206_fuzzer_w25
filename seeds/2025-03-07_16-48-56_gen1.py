import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensors):
        results = []
        for tensor in input_tensors:
            # Perform a simple operation such as element-wise multiplication by 2
            results.append(tensor * 2)
        return results

input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]),  # 4D tensor
    torch.tensor([[9, 10, 11, 12], [13, 14, 15, 16]]),  # 2D tensor
    torch.tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    for i, o in enumerate(output):
        print(f"Output {i}: {o}")