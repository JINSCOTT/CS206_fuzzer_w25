import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        add_result = x + 2
        sub_result = x - 2
        mul_result = x * 2
        div_result = x / 2
        eq_result = (x == 2)
        lt_result = (x < 2)
        gt_result = (x > 2)
        
        return add_result, sub_result, mul_result, div_result, eq_result, lt_result, gt_result

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),    # 2D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1], [2]], [[3], [4]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),    # another 2D tensor
    torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)      # another 2D tensor (column)
]

def main():
    model = PtModule()
    for input_tensor in input_tensors:
        results = model(input_tensor)
        print(f"Input: {input_tensor}")
        print(f"Results: {results}")

if __name__ == "__main__":
    main()