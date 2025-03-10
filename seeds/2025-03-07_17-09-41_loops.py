import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
    
    def forward(self, x):
        # Perform various operations and loops
        result = []
        for i in range(x.size(0)):
            temp = x[i] * 2  # Multiply each element by 2
            temp = temp + 3  # Add 3 to each element
            result.append(temp)
        
        result = torch.stack(result)  # Stack the results into a tensor
        return result

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[9, 10], [11, 12]]], dtype=torch.float32),
    torch.tensor([[[13], [14]], [[15], [16]], [[17], [18]]], dtype=torch.float32),
    torch.tensor([[[19, 20, 21], [22, 23, 24]]], dtype=torch.float32),
    torch.tensor([[[25], [26], [27]], [[28], [29], [30]]], dtype=torch.float32)
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    module = PtModule()
    for input_tensor in input_tensors:
        output = module(input_tensor)
        print(output)