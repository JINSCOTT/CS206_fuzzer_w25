import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        # Addition
        add_result = inputs[0] + inputs[1]
        # Subtraction
        sub_result = inputs[2] - inputs[3]
        # Multiplication
        mul_result = inputs[1] * inputs[4]
        # Division
        div_result = inputs[4] / (inputs[2] + 1e-6)  # Adding small epsilon to avoid division by zero
        # Comparison
        comparison_result = inputs[0] > inputs[1]
        
        return add_result, sub_result, mul_result, div_result, comparison_result

# Defining input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([5.0, 6.0]),                 # 1D tensor
    torch.tensor([2.0, 3.0, 4.0]),            # 1D tensor
    torch.tensor([[1.0, 0.0], [0.0, 1.0]]),  # 2D tensor
    torch.tensor([10.0, 20.0, 30.0, 40.0])    # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    results = model(input_tensors)
    for result in results:
        print(result)