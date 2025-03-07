import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = input_tensor.clone()
        for i in range(result.shape[0]):
            result[i] = result[i] * 2 + 3  # Example operation
        return result

input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]], [[[5.0], [6.0]], [[7.0], [8.0]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=torch.float32)  # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")