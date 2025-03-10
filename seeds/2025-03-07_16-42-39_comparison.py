import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        addition = inputs[0] + inputs[1]
        subtraction = inputs[2] - inputs[3]
        multiplication = inputs[1] * inputs[2]
        division = inputs[3] / (inputs[4] + 1e-8)  # adding a small value to avoid division by zero
        comparison = inputs[0] > inputs[4]
        
        return addition, subtraction, multiplication, division, comparison

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[5], [6]], [[7], [8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[9, 10, 11], [12, 13, 14]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[15, 16], [17, 18]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[19], [20]], [[21], [22]]], dtype=torch.float32)   # 3D tensor
]

if __name__ == "__main__":
    module = PtModule()
    results = module(input_tensors)
    for res in results:
        print(res)