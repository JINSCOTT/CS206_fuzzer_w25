import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2):
        add_result = x1 + x2
        sub_result = x1 - x2
        mul_result = x1 * x2
        div_result = x1 / (x2 + 1e-8)  # preventing division by zero
        greater_result = x1 > x2
        less_result = x1 < x2
        return add_result, sub_result, mul_result, div_result, greater_result, less_result

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[1, 2], [3, 4]]),  # 2D tensor with integers
    torch.tensor([1, 2, 3, 4, 5])  # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    results = model(input_tensors[0], input_tensors[0])  # Using the first tensor for demonstration
    print("Results:", results)