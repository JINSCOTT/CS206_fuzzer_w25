import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, tensors):
        results = {}
        # Math operations
        results['add'] = tensors[0] + tensors[1]
        results['subtract'] = tensors[0] - tensors[1]
        results['multiply'] = tensors[0] * tensors[1]
        results['divide'] = tensors[0] / tensors[1]
        results['power'] = tensors[0] ** 2
        
        # Comparison operations
        results['greater'] = tensors[0] > tensors[1]
        results['less'] = tensors[0] < tensors[1]
        results['equal'] = tensors[0] == tensors[1]
        results['greater_equal'] = tensors[0] >= tensors[1]
        results['less_equal'] = tensors[0] <= tensors[1]
        
        return results

# Input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]).float(),
    torch.tensor([[6, 5, 4], [3, 2, 1]]).float(),
    torch.tensor([[1, 1, 1], [1, 1, 1]]).float(),
    torch.tensor([[2, 2, 2], [2, 2, 2]]).float(),
    torch.tensor([[3, 3, 3], [3, 3, 3]]).float()
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    for key, value in outputs.items():
        print(f"{key}:\n{value}\n")