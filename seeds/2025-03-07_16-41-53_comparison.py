import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Applying mathematical operations
        add_result = x + 2
        sub_result = x - 2
        mul_result = x * 2
        div_result = x / 2
        
        # Applying comparison operations
        greater_than = x > 1
        less_than = x < 3
        equal_to = x == 2
        
        return add_result, sub_result, mul_result, div_result, greater_than, less_than, equal_to

# Defining input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[2, 4, 6], [8, 10, 12]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]], dtype=torch.float32),
    torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float32)
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    module = PtModule()
    for input_tensor in input_tensors:
        results = module(input_tensor)
        print(f"Results for input tensor:\n{input_tensor}\n{results}\n")