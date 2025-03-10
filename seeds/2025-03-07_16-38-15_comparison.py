import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations and comparisons
        output = {}
        
        # Math operations
        output['addition'] = x + 5
        output['subtraction'] = x - 3
        output['multiplication'] = x * 2
        output['division'] = x / 2
        
        # Comparison operations
        output['greater_than'] = x > 2
        output['less_than'] = x < 5
        output['equal_to'] = x == 3
        
        return output

input_tensors = [
    torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32),
    torch.tensor([[[5, 6, 7], [8, 9, 10]]], dtype=torch.float32),
    torch.tensor([[[11], [12]], [[13], [14]]], dtype=torch.float32),
    torch.tensor([[[15, 16]], [[17, 18]], [[19, 20]]], dtype=torch.float32),
    torch.tensor([[[21, 22], [23, 24], [25, 26]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)