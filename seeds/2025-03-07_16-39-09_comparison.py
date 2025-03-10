import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = {}
        results['addition'] = inputs[0] + inputs[1]
        results['subtraction'] = inputs[0] - inputs[1]
        results['multiplication'] = inputs[0] * inputs[1]
        results['division'] = inputs[0] / (inputs[1] + 1e-8)  # Adding small value to avoid division by zero
        results['greater_than'] = inputs[0] > inputs[1]
        results['less_than'] = inputs[0] < inputs[1]
        results['equal_to'] = inputs[0] == inputs[1]
        return results

# Input Tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[2.0, 1.0], [0.0, 5.0]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32),        # 3D tensor
    torch.tensor([[[[2], [3]], [[4], [5]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([1, 2, 3, 4], dtype=torch.float32)                # 1D tensor
]

if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)