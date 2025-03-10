import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        # Perform various operations
        addition = inputs[0] + inputs[1]
        subtraction = inputs[1] - inputs[2]
        multiplication = inputs[2] * inputs[3]
        division = inputs[3] / (inputs[4] + 1e-5)  # Avoid division by zero
        comparison = inputs[0] > inputs[4]

        return addition, subtraction, multiplication, division, comparison

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D Tensor
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),  # 3D Tensor
    torch.tensor([[[[9.0, 10.0]], [[11.0, 12.0]]]]),  # 4D Tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D Tensor
    torch.tensor([[0.0, 1.0], [2.0, 3.0]])   # 2D Tensor
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    results = model(input_tensors)
    print("Results:")
    for res in results:
        print(res)