import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        result = []
        for input_tensor in inputs:
            # Performing normal math operations
            added = input_tensor + 2
            subtracted = input_tensor - 1
            multiplied = input_tensor * 3
            divided = input_tensor / 2
            result.append((added, subtracted, multiplied, divided))
        return result

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),  # 4D tensor
    torch.tensor([[7, 8, 9]]),  # 2D tensor
    torch.tensor([[[10], [11]], [[12], [13]]])  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output for input tensor {i}:\n{output}\n")