import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Math Operations
        add_result = input_tensor + 2
        sub_result = input_tensor - 3
        mul_result = input_tensor * 4
        div_result = input_tensor / 5

        # Comparison Operations
        greater_result = input_tensor > 1
        less_result = input_tensor < 5
        equal_result = input_tensor == 2

        return add_result, sub_result, mul_result, div_result, greater_result, less_result, equal_result

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),
    torch.tensor([[1], [2], [3], [4]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32),
    torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        results = model(tensor)
        print("Results for input tensor:\n", tensor)
        for result in results:
            print(result)