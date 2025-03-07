import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Perform some math operations: addition, multiplication, and exponentiation
        result = input_tensor + 5
        result = result * 2
        result = result ** 2
        return result

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[1], [2], [3], [4]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1]]], dtype=torch.float32)  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")