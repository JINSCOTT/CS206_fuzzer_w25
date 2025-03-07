import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        add_result = x + 2
        sub_result = x - 2
        mul_result = x * 2
        div_result = x / 2
        greater_than_result = x > 2
        less_than_result = x < 2
        return add_result, sub_result, mul_result, div_result, greater_than_result, less_than_result

input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[1.5, 2.5], [3.5, 4.5]]),  # 2D tensor
    torch.tensor([[[[2.0, 3.0]], [[4.0, 5.0]]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        results = model(input_tensor)
        print(f"Results for input tensor {i+1}: {results}")