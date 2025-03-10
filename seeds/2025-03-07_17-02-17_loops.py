import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            multiplied = input_tensor * 2
            added = multiplied + 3
            results.append(added)
        return results

input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),  # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]]]),  # 3D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),  # 2D tensor
    torch.tensor([[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]]),  # 4D tensor
    torch.tensor([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output for input tensor {i}: {output}")