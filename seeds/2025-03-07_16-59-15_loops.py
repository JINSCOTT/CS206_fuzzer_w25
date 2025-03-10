import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Performing some math operations
        result = input_tensor + 2
        
        # Looping through the dimensions
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i][j] = result[i][j] * 3

        return result

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32),
    torch.tensor([[[[1], [2]], [[3], [4]]]], dtype=torch.float32),
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
    torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32),
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)