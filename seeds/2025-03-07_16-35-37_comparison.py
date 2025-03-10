import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x1, x2, x3):
        # Math operators
        addition = x1 + x2
        subtraction = x1 - x2
        multiplication = x1 * x2
        division = x1 / (x2 + 1e-5)  # Avoiding division by zero
        
        # Comparison operators
        greater_than = x1 > x3
        less_than = x1 < x3
        equal_to = x1 == x3

        return addition, subtraction, multiplication, division, greater_than, less_than, equal_to

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[1.0, 0.0], [0.0, 1.0]]),  # 2D tensor
]

def main():
    model = PtModule()
    results = model(input_tensors[0], input_tensors[1].squeeze(), input_tensors[2])
    for result in results:
        print(result)

if __name__ == "__main__":
    main()