import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2):
        addition = x1 + x2
        subtraction = x1 - x2
        multiplication = x1 * x2
        division = x1 / (x2 + 1e-6)  # Adding small value to avoid division by zero
        comparison = x1 > x2
        
        return addition, subtraction, multiplication, division, comparison

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[9.0, 10.0], [11.0, 12.0]]]),  # 3D tensor
    torch.tensor([[[13.0, 14.0], [15.0, 16.0]]]),  # 3D tensor
    torch.tensor([[17.0, 18.0], [19.0, 20.0]])    # 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors[0], input_tensors[1])
    print("Addition:\n", outputs[0])
    print("Subtraction:\n", outputs[1])
    print("Multiplication:\n", outputs[2])
    print("Division:\n", outputs[3])
    print("Comparison:\n", outputs[4])