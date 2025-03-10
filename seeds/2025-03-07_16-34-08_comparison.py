import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Applying various math operations
        add_result = input_tensor + 10
        subtract_result = input_tensor - 5
        multiply_result = input_tensor * 2
        divide_result = input_tensor / 3.0

        # Applying comparison operators
        greater_than = input_tensor > 5
        less_than = input_tensor < 2

        return add_result, subtract_result, multiply_result, divide_result, greater_than, less_than

input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),                    # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[10.0, 11.0], [12.0, 13.0]]),                # Another 2D tensor
    torch.tensor([1.0, 2.0, 3.0])                              # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        results = model(tensor)
        print("Input Tensor:\n", tensor)
        print("Results:\n", results)