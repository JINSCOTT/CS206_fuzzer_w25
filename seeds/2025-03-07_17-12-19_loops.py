import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            result = input_tensor * 2  # Multiply by 2
            result = result + 3        # Add 3
            result = result - 1        # Subtract 1
            result = result / 2        # Divide by 2
            results.append(result)
        return results

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),  # 4D tensor
    torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),  # 2D tensor
    torch.tensor([[[[11.0], [12.0]], [[13.0], [14.0]], [[15.0], [16.0]]]])  # 4D tensor
]

if __name__ == "__main__":
    module = PtModule()
    outputs = module(input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output for input tensor {i}:")
        print(output)