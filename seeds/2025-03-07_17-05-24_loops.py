import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = tensor + 2  # Addition
            result = result * 3  # Multiplication
            result = torch.mean(result)  # Mean
            results.append(result)
        return results

# Defining input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32),  # 3D Tensor
    torch.tensor([[[5, 6, 7], [8, 9, 10]]], dtype=torch.float32),  # 3D Tensor
    torch.tensor([[[11, 12], [13, 14]], [[15, 16], [17, 18]]], dtype=torch.float32),  # 4D Tensor
    torch.tensor([[[19], [20]], [[21], [22]], [[23], [24]]], dtype=torch.float32),  # 4D Tensor
    torch.tensor([[25, 26, 27], [28, 29, 30]], dtype=torch.float32)  # 2D Tensor
]

def main():
    model = PtModule()
    outputs = model(input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output for input tensor {i+1}: {output.item()}")

if __name__ == "__main__":
    main()