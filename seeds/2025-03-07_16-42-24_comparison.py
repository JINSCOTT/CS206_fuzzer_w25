import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = {}
        results['add'] = inputs[0] + inputs[1]
        results['subtract'] = inputs[0] - inputs[1]
        results['multiply'] = inputs[0] * inputs[1]
        results['divide'] = inputs[0] / (inputs[1] + 1e-8)  # Avoid division by zero
        results['greater_than'] = inputs[0] > inputs[1]
        results['less_than'] = inputs[0] < inputs[1]
        results['equal'] = inputs[0] == inputs[1]
        return results

input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),                # 2D tensor
    torch.tensor([[1.0, 1.0], [1.0, 1.0]]),                # 2D tensor
    torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),           # 3D tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),       # 4D tensor
    torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]]) # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    results = model(input_tensors[:2])  # Use the first two tensors for operations
    print("Results:", results)