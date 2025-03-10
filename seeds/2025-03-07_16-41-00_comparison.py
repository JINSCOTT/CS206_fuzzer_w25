import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x, y):
        addition = x + y
        subtraction = x - y
        multiplication = x * y
        division = x / (y + 1e-6)  # Adding a small value to avoid division by zero
        equality = (x == y).float()
        greater_than = (x > y).float()
        less_than = (x < y).float()
        
        return addition, subtraction, multiplication, division, equality, greater_than, less_than

input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[9.0, 10.0], [11.0, 12.0]]]),  # 3D tensor
    torch.tensor([[[13.0, 14.0], [15.0, 16.0]]]),  # 3D tensor
    torch.tensor([[[17.0, 18.0], [19.0, 20.0]]])   # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for i in range(len(input_tensors) - 1):
        results = model(input_tensors[i], input_tensors[i + 1])
        print(f"Results for input tensors {i} and {i + 1}:")
        for result in results:
            print(result)