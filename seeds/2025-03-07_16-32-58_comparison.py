import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example math operations
        add_result = x + 2
        sub_result = x - 2
        mul_result = x * 2
        div_result = x / 2
        
        # Example comparison operations
        greater_than = x > 1
        less_than = x < 3
        
        return add_result, sub_result, mul_result, div_result, greater_than, less_than

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),  # 2D tensor
    torch.tensor([[[[9.0, 10.0], [11.0, 12.0]]]])  # 4D tensor
]

def main():
    model = PtModule()
    for input_tensor in input_tensors:
        add_result, sub_result, mul_result, div_result, greater_than, less_than = model(input_tensor)
        print("Input:\n", input_tensor)
        print("Add Result:\n", add_result)
        print("Sub Result:\n", sub_result)
        print("Mul Result:\n", mul_result)
        print("Div Result:\n", div_result)
        print("Greater Than 1:\n", greater_than)
        print("Less Than 3:\n", less_than)
        print("-" * 30)

if __name__ == "__main__":
    main()