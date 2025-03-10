import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Apply some math operations and loops
        output = input_tensor * 2  # Multiply by 2
        for i in range(5):
            output = output + i  # Add i (from 0 to 4) to the output
        output = output / 5  # Average the result
        return output

# Input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[[10.0, 20.0], [30.0, 40.0]]]),  # 3D tensor
    torch.tensor([[[[1.0]]]]),  # 4D tensor
    torch.tensor([[[5.0, 10.0, 15.0], [20.0, 25.0, 30.0]], [[35.0, 40.0, 45.0], [50.0, 55.0, 60.0]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Input:\n{tensor}\nOutput:\n{output}\n")