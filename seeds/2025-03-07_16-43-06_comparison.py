import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Perform various math operations
        addition = x + 5
        subtraction = x - 3
        multiplication = x * 2
        division = x / 4
        power = x ** 2
        
        # Perform comparisons
        greater_than = x > 2
        less_than = x < 4
        equal_to = x == 1
        
        # Return a dictionary of results
        return {
            'addition': addition,
            'subtraction': subtraction,
            'multiplication': multiplication,
            'division': division,
            'power': power,
            'greater_than': greater_than,
            'less_than': less_than,
            'equal_to': equal_to
        }

# Define input tensors
input_tensors = [
    torch.tensor([1, 2, 3], dtype=torch.float32),       # 1D tensor
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32), # 4D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32) # 2D tensor with 2 rows
]

def main():
    model = PtModule()
    for idx, input_tensor in enumerate(input_tensors):
        print(f"Input Tensor {idx+1}:")
        output = model(input_tensor)
        for key, value in output.items():
            print(f"{key}: \n{value}\n")

if __name__ == "__main__":
    main()