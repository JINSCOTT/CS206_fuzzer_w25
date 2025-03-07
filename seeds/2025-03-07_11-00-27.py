import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Applying various math operations
        add_result = input_tensor + 10
        sub_result = input_tensor - 5
        mul_result = input_tensor * 2
        div_result = input_tensor / 2
        # Applying comparisons
        greater_than = input_tensor > 5
        less_than = input_tensor < 10
        
        return add_result, sub_result, mul_result, div_result, greater_than, less_than

# Define input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),          # 2D tensor
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),        # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),      # 4D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # 4D tensor
]

def main():
    model = PtModule()
    for input_tensor in input_tensors:
        add, sub, mul, div, gt, lt = model(input_tensor)
        print("Input Tensor:\n", input_tensor)
        print("Add Result:\n", add)
        print("Subtract Result:\n", sub)
        print("Multiply Result:\n", mul)
        print("Divide Result:\n", div)
        print("Greater Than 5:\n", gt)
        print("Less Than 10:\n", lt)
        print("--------------------------------")

if __name__ == "__main__":
    main()