import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Example math operations
        output = input_tensor * 2  # Scale by 2
        for i in range(input_tensor.size(0)):
            output[i] = output[i] + i  # Add index to each element
        return output

# Defining input tensors
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),
    torch.tensor([[[13, 14], [15, 16]], [[17, 18], [19, 20]]], dtype=torch.float32),
    torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32),
    torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32),
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output_tensor = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output_tensor}\n")