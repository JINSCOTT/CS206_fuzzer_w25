import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
    
    def forward(self, x):
        results = {}
        # Mathematical operations
        results['add'] = x + 2
        results['subtract'] = x - 2
        results['multiply'] = x * 2
        results['divide'] = x / 2
        
        # Comparisons
        results['greater_than'] = x > 1
        results['less_than'] = x < 3
        results['equal_to'] = x == 2
        
        return results

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D Tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D Tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D Tensor
    torch.tensor([[1.5, 2.5], [3.5, 4.5]]),  # 2D Tensor with float values
    torch.tensor([1, 2, 3, 4, 5])  # 1D Tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")