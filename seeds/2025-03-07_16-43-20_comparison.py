import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2):
        results = {}
        results['addition'] = x1 + x2
        results['subtraction'] = x1 - x2
        results['multiplication'] = x1 * x2
        results['division'] = x1 / (x2 + 1e-10)  # Adding small value to avoid division by zero
        results['greater_than'] = x1 > x2
        results['less_than'] = x1 < x2
        results['equal_to'] = x1 == x2
        return results

input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor
    torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]]]),  # 4D tensor
    torch.tensor([[1.0], [0.0], [1.0]])  # 2D tensor with a different shape
]

if __name__ == "__main__":
    model = PtModule()
    x1 = input_tensors[0]
    x2 = input_tensors[1]
    output = model(x1, x2)
    print(output)