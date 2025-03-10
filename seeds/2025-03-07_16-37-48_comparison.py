import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = {}
        # Addition
        results['addition'] = inputs[0] + inputs[1]
        # Subtraction
        results['subtraction'] = inputs[0] - inputs[1]
        # Multiplication
        results['multiplication'] = inputs[0] * inputs[1]
        # Division
        results['division'] = inputs[0] / (inputs[1] + 1e-7)  # Adding small value to avoid division by zero
        # Comparison
        results['greater_than'] = inputs[0] > inputs[1]
        results['less_than'] = inputs[0] < inputs[1]
        results['equal'] = inputs[0] == inputs[1]
        return results

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor
    torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])   # Another 2D tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)