import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Add two tensors
        add_result = input_tensor[0] + input_tensor[1]
        # Subtract two tensors
        sub_result = input_tensor[1] - input_tensor[2]
        # Multiply two tensors
        mul_result = input_tensor[2] * input_tensor[3]
        # Divide two tensors
        div_result = input_tensor[3] / (input_tensor[4] + 1e-5)  # Avoid division by zero
        # Compare two tensors
        comparison_result = input_tensor[0] > input_tensor[4]

        return add_result, sub_result, mul_result, div_result, comparison_result

input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[5, 6], [7, 8]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[1, 1], [1, 1]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[2, 3], [4, 5]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)   # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    results = model(input_tensors)
    for result in results:
        print(result)