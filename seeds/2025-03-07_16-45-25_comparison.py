import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        # Adding inputs
        add_result = inputs[0] + inputs[1]
        # Subtracting inputs
        sub_result = inputs[2] - inputs[3]
        # Multiplying inputs
        mul_result = inputs[0] * inputs[4]
        # Dividing inputs
        div_result = inputs[3] / (inputs[1] + 1e-5)  # Avoiding division by zero
        # Comparing inputs
        comp_result = inputs[2] > inputs[0]

        return add_result, sub_result, mul_result, div_result, comp_result

# Input Tensors defined outside the module
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 3D Tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),  # 3D Tensor
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 4D Tensor
    torch.tensor([[2.0, 3.0], [4.0, 5.0]]),  # 3D Tensor
    torch.tensor([[0.5, 1.0], [1.5, 2.0]])   # 3D Tensor
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    results = model(input_tensors)
    for result in results:
        print(result)