import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        results = {}
        results['addition'] = x + 2
        results['multiplication'] = x * 3
        results['subtraction'] = x - 1
        results['division'] = x / 4
        results['greater_than'] = x > 1
        results['less_than'] = x < 5
        return results

# Defining input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), # 4D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6]]), # 2D tensor
    torch.tensor([1.0, 2.0, 3.0, 4.0]), # 1D tensor
    torch.tensor([[[1], [2]], [[3], [4]]]) # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input Tensor:\n{tensor}\nOutput:\n{output}\n")